import asyncio
from typing import List, Optional

import aiohttp
from pydantic.config import Extra
from pydantic.fields import Field
from pydantic.main import BaseModel

from surfer.common import constants
from surfer.common.exceptions import InternalError
from surfer.computing import schemas
from surfer.computing.services import PricingService
from surfer.log import logger

RETAIL_PRICES_API_VERSION = "2023-01-01-preview"
RETAIL_PRICES_API_URL = (
    "https://prices.azure.com/api/retail/prices?api-version={}".format(
        RETAIL_PRICES_API_VERSION,
    )
)


class _RetailPriceRespItem(BaseModel):
    price: float = Field(..., alias="retailPrice")
    sku: str = Field(..., alias="skuName")
    region: str = Field(..., alias="armRegionName")
    currency: str = Field(..., alias="currencyCode")
    product_name: str = Field(..., alias="productName")
    reservation_term: Optional[str] = Field(None, alias="reservationTerm")

    class Config:
        extras = Extra.ignore

    def is_windows(self) -> bool:
        return "windows" in self.product_name.lower()


class _RetailPricesResp(BaseModel):
    items: List[_RetailPriceRespItem] = Field(..., alias="Items")
    currency: str = Field(..., alias="BillingCurrency")

    class Config:
        extras = Extra.ignore

    @property
    def linux_items(self) -> List[_RetailPriceRespItem]:
        return [item for item in self.items if not item.is_windows()]


class _RetailPricesQuery:
    def __init__(self):
        self._query = "$filter=unitOfMeasure eq '1 Hour' "

    def with_currency(self, currency_code: str) -> "_RetailPricesQuery":
        self._query += " and currencyCode eq '{}'".format(currency_code)
        return self

    def with_region(self, region: str) -> "_RetailPricesQuery":
        self._query += " and armRegionName eq '{}'".format(region)
        return self

    def with_sku(self, sku: str) -> "_RetailPricesQuery":
        self._query += " and armSkuName eq '{}'".format(sku)
        return self

    def with_spot_priority(self) -> "_RetailPricesQuery":
        self._query += " and contains(skuName, 'Spot')"
        return self

    def with_price_type(self, t: str) -> "_RetailPricesQuery":
        """
        Parameters
        ----------
        t: str
            Allowed values are "Consumption" or "Reservation".
        """
        self._query += " and priceType eq '{}'".format(t)
        return self

    def get_query(self) -> str:
        return self._query


class _RetailPricingClient:
    def __init__(self, url: str, sku: str, region: str, currency: str):
        self._url = url
        self._sku = sku
        self._region = region
        self._currency = currency
        # Responses
        self._consumption_resp: Optional[_RetailPriceRespItem] = None
        self._spot_resp: Optional[_RetailPriceRespItem] = None
        self._discounted_1yr_resp: Optional[_RetailPriceRespItem] = None
        self._discounted_3yr_resp: Optional[_RetailPriceRespItem] = None

    def _get_url(self, query: str):
        return "{}&{}".format(self._url, query)

    async def __call__(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> _RetailPricesResp:
        logger.debug("GET > {}".format(url))
        async with session.get(url) as resp:
            resp.raise_for_status()
            resp_dict = await resp.json()
        return _RetailPricesResp.parse_obj(resp_dict)

    async def _fetch_discount_pricing(self, session: aiohttp.ClientSession):
        reservation = (
            _RetailPricesQuery()
            .with_price_type("Reservation")
            .with_sku(self._sku)
            .with_region(self._region)
            .with_currency(self._currency)
            .get_query()
        )
        resp = await self(session, self._get_url(reservation))
        linux_vms = resp.linux_items
        if len(linux_vms) == 0:
            logger.warn(
                "no reservation pricing found for {} - {}".format(
                    self._sku,
                    self._region,
                ),
            )
            return
        for vm in linux_vms:
            if vm.reservation_term == "1 Year":
                self._discounted_1yr_resp = vm
            elif vm.reservation_term == "3 Years":
                self._discounted_3yr_resp = vm
        if self._discounted_1yr_resp is None:
            logger.warn(
                "no 1 year reservation pricing found for {} - {}".format(
                    self._sku,
                    self._region,
                ),
            )
        if self._discounted_3yr_resp is None:
            logger.warn(
                "no 3 year reservation pricing found for {} - {}".format(
                    self._sku,
                    self._region,
                ),
            )

    async def _fetch_consumption_pricing(self, session: aiohttp.ClientSession):
        def __sku_is_spot(i: _RetailPriceRespItem) -> bool:
            if "spot" in i.sku.lower():
                return True
            if "priority" in i.sku.lower():
                return True
            return False

        query = (
            _RetailPricesQuery()
            .with_price_type("Consumption")
            .with_sku(self._sku)
            .with_region(self._region)
            .with_currency(self._currency)
            .get_query()
        )
        resp = await self(session, self._get_url(query))
        linux_vms = list(
            filter(lambda x: not __sku_is_spot(x), resp.linux_items),
        )
        if len(linux_vms) == 0:
            logger.warn(
                "no consumption price found for {} - {}".format(
                    self._sku,
                    self._region,
                ),
            )
            return
        if len(linux_vms) > 1:
            logger.warn(
                "more than one consumption price found for {} - {}".format(
                    self._sku,
                    self._region,
                )
            )
        self._consumption_resp = linux_vms[0]

    async def _fetch_spot_pricing(self, session: aiohttp.ClientSession):
        query = (
            _RetailPricesQuery()
            .with_price_type("Consumption")
            .with_sku(self._sku)
            .with_spot_priority()
            .with_region(self._region)
            .with_currency(self._currency)
            .get_query()
        )
        resp = await self(session, self._get_url(query))
        linux_vms = resp.linux_items
        if len(linux_vms) == 0:
            logger.warn(
                "no spot price found for {} - {}".format(
                    self._sku,
                    self._region,
                ),
            )
            return
        if len(linux_vms) > 1:
            logger.warn(
                "more than one spot price found for {} - {}".format(
                    self._sku,
                    self._region,
                )
            )
        self._spot_resp = linux_vms[0]

    async def get_pricing(
        self,
        session: aiohttp.ClientSession,
    ) -> schemas.VMPricing:
        try:
            await asyncio.gather(
                self._fetch_consumption_pricing(session),
                self._fetch_spot_pricing(session),
                self._fetch_discount_pricing(session),
            )
            price_hr = None
            spot_price_hr = None
            price_hr_1yr = None
            price_hr_3yr = None
            if self._consumption_resp is not None:
                price_hr = self._consumption_resp.price
            if self._spot_resp is not None:
                spot_price_hr = self._spot_resp.price
            if self._discounted_1yr_resp is not None:
                hours = constants.HOURS_PER_YEAR
                price_hr_1yr = self._discounted_1yr_resp.price / hours
            if self._discounted_3yr_resp is not None:
                hours = constants.HOURS_PER_YEAR * 3
                price_hr_3yr = self._discounted_3yr_resp.price / hours
            return schemas.VMPricing(
                sku=self._sku,
                region=self._region,
                currency=self._currency,
                price_hr=price_hr,
                price_hr_spot=spot_price_hr,
                price_hr_1yr=price_hr_1yr,
                price_hr_3yr=price_hr_3yr,
            )
        except Exception as e:
            raise InternalError(
                "failed to fetch pricing info for {} - {}: {}".format(
                    self._sku,
                    self._region,
                    e,
                )
            )


class AzurePricingService(PricingService):
    def __init__(self, api_url: str = RETAIL_PRICES_API_URL):
        self.api_url = api_url

    async def get_vm_pricing(
        self,
        vm_sku: str,
        region: str,
        currency: str = "USD",
        **kwargs,
    ) -> schemas.VMPricing:
        async with aiohttp.ClientSession() as session:
            client = _RetailPricingClient(
                url=self.api_url,
                sku=vm_sku,
                region=region,
                currency=currency,
            )
            return await client.get_pricing(session)
