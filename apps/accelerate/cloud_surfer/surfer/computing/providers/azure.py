import asyncio
from typing import List

import aiohttp
from pydantic.config import Extra
from pydantic.fields import Field
from pydantic.main import BaseModel

from surfer.common.exceptions import NotFoundError
from surfer.computing.models import VMPricingInfo
from surfer.computing.services import PricingService
from surfer.log import logger, configure_debug_mode

RETAIL_PRICES_API_VERSION = "2023-01-01-preview"
RETAIL_PRICES_API_URL = (
    "https://prices.azure.com/api/retail/prices?api-version={}".format(
        RETAIL_PRICES_API_VERSION,
    )
)


class _RetainPriceRespItem(BaseModel):
    price: float = Field(..., alias="retailPrice")
    sku: str = Field(..., alias="skuName")
    region: str = Field(..., alias="armRegionName")
    currency: str = Field(..., alias="currencyCode")
    product_name: str = Field(..., alias="productName")

    class Config:
        extras = Extra.ignore

    def is_windows(self) -> bool:
        return "windows" in self.product_name.lower()


class _RetailPricesResp(BaseModel):
    items: List[_RetainPriceRespItem] = Field(..., alias="Items")
    currency: str = Field(..., alias="BillingCurrency")

    class Config:
        extras = Extra.ignore

    @property
    def linux_items(self) -> List[_RetainPriceRespItem]:
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


class AzurePricingService(PricingService):
    def __init__(self, api_url: str = RETAIL_PRICES_API_URL):
        self.api_url = api_url

    def _get_url(self, query: str):
        return "{}&{}".format(self.api_url, query)

    async def _get_spot_price(
        self,
        session: aiohttp.ClientSession,
        sku: str,
        region: str,
        currency: str,
    ) -> float:
        consumption = (
            _RetailPricesQuery()
            .with_price_type("Consumption")
            .with_sku(sku)
            .with_region(region)
            .with_currency(currency)
            .get_query()
        )
        reservation = (
            _RetailPricesQuery()
            .with_price_type("Reservation")
            .with_sku(sku)
            .with_region(region)
            .with_currency(currency)
            .get_query()
        )
        query = (
            _RetailPricesQuery()
            .with_price_type("Consumption")
            .with_sku(sku)
            .with_spot_priority()
            .with_region(region)
            .with_currency(currency)
            .get_query()
        )
        url = self._get_url(query)
        logger.debug("GET > {}".format(url))
        async with session.get(url) as resp:
            resp.raise_for_status()
            resp_dict = await resp.json()
        resp_obj = _RetailPricesResp.parse_obj(resp_dict)
        linux_vms = resp_obj.linux_items
        if len(linux_vms) == 0:
            raise NotFoundError(
                "no spot price found for {} - {}".format(
                    sku,
                    region,
                ),
            )
        if len(linux_vms) > 1:
            logger.warn(
                "more than one spot price found for {} - {}".format(
                    sku,
                    region,
                )
            )
        return linux_vms[0].price

    async def get_vm_pricing(
        self,
        vm_size: str,
        region: str,
        currency: str = "USD",
        **kwargs,
    ) -> VMPricingInfo:
        async with aiohttp.ClientSession() as session:
            return await self._get_spot_price(
                session,
                sku=vm_size,
                region=region,
                currency=currency,
            )


async def main():
    configure_debug_mode(True)
    service = AzurePricingService()
    res = await service.get_vm_pricing(
        vm_size="Standard_D2s_v3",
        region="eastus",
    )
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
