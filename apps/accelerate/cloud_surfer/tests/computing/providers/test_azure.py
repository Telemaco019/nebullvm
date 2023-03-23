import unittest
from unittest.mock import patch, MagicMock

from surfer.common import constants
from surfer.common.exceptions import InternalError
from surfer.computing.models import VMPricingInfo
from surfer.computing.providers.azure import (
    AzurePricingService,
    _RetailPricingClient,
    _RetailPriceRespItem,
)


class TestAzurePricingService(unittest.IsolatedAsyncioTestCase):
    @patch.object(_RetailPricingClient, "get_pricing_info")
    async def test_get_vm_pricing(self, mock):
        service = AzurePricingService()
        info = VMPricingInfo(
            "USD",
            "westus",
            "Standard_D2s_v3",
            0.0,
            0.0,
            0.0,
            0.0,
        )
        mock.return_value = info
        res = await service.get_vm_pricing(
            vm_size="Standard_D2s_v3",
            region="westus",
            currency="USD",
        )
        self.assertEqual(info, res)


class TestRetailPricingClient(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_discount_pricing__empty(self):
        client = _RetailPricingClient(
            "",
            "Standard_D2s_v3",
            "westus",
            "USD",
        )
        session = MagicMock()
        session.get.return_value.__aenter__.return_value.json.return_value = {
            "Items": [],
            "BillingCurrency": "USD",
        }
        await client._fetch_discount_pricing(session)
        self.assertIsNone(client._discounted_1yr_resp)
        self.assertIsNone(client._discounted_3yr_resp)

    async def test_fetch_discount_pricing__no_reservation_info(self):
        client = _RetailPricingClient(
            "",
            "Standard_D2s_v3",
            "westus",
            "USD",
        )
        session = MagicMock()
        session.get.return_value.__aenter__.return_value.json.return_value = {
            "Items": [
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="Standard_D2s_v3",
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3",
                    reservationTerm=None,
                ).dict(by_alias=True),
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="3yr",
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3 Windows",  # should be ignored
                    reservationTerm="3 Years",
                ).dict(by_alias=True),
            ],
            "BillingCurrency": "USD",
        }
        await client._fetch_discount_pricing(session)
        self.assertIsNone(client._discounted_1yr_resp)
        self.assertIsNone(client._discounted_3yr_resp)

    async def test_fetch_discount_pricing__reservation_info(self):
        client = _RetailPricingClient(
            "",
            "Standard_D2s_v3",
            "westus",
            "USD",
        )
        session = MagicMock()
        session.get.return_value.__aenter__.return_value.json.return_value = {
            "Items": [
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="1yr",
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3",
                    reservationTerm="1 Year",
                ).dict(by_alias=True),
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="3yr",
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3",
                    reservationTerm="3 Years",
                ).dict(by_alias=True),
            ],
            "BillingCurrency": "USD",
        }
        await client._fetch_discount_pricing(session)
        self.assertIsNotNone(client._discounted_1yr_resp)
        self.assertEqual(client._discounted_1yr_resp.sku, "1yr")
        self.assertIsNotNone(client._discounted_3yr_resp)
        self.assertEqual(client._discounted_3yr_resp.sku, "3yr")

    async def test_fetch_spot_pricing__no_results(self):
        client = _RetailPricingClient(
            "",
            "Standard_D2s_v3",
            "westus",
            "USD",
        )
        session = MagicMock()
        session.get.return_value.__aenter__.return_value.json.return_value = {
            "Items": [
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="3yr",
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3 Windows",  # should be ignored
                    reservationTerm=None,
                ).dict(by_alias=True),
            ],
            "BillingCurrency": "USD",
        }
        await client._fetch_spot_pricing(session)
        self.assertIsNone(client._spot_resp)

    async def test_fetch_spot_pricing__multiple_results(self):
        client = _RetailPricingClient(
            "",
            "Standard_D2s_v3",
            "westus",
            "USD",
        )
        session = MagicMock()
        session.get.return_value.__aenter__.return_value.json.return_value = {
            "Items": [
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="sku-1",
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3",
                    reservationTerm=None,
                ).dict(by_alias=True),
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="sku-2",
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3",
                    reservationTerm=None,
                ).dict(by_alias=True),
            ],
            "BillingCurrency": "USD",
        }
        await client._fetch_spot_pricing(session)
        self.assertIsNotNone(client._spot_resp)
        self.assertEqual(client._spot_resp.sku, "sku-1")

    async def test_fetch_consumption_pricing__multiple_results(self):
        client = _RetailPricingClient(
            "",
            "Standard_D2s_v3",
            "westus",
            "USD",
        )
        session = MagicMock()
        session.get.return_value.__aenter__.return_value.json.return_value = {
            "Items": [
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="sku-1",
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3",
                    reservationTerm=None,
                ).dict(by_alias=True),
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="sku-2",
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3",
                    reservationTerm=None,
                ).dict(by_alias=True),
            ],
            "BillingCurrency": "USD",
        }
        await client._fetch_consumption_pricing(session)
        self.assertIsNotNone(client._consumption_resp)
        self.assertEqual(client._consumption_resp.sku, "sku-1")

    async def test_fetch_consumption_pricing__no_results(self):
        client = _RetailPricingClient(
            "",
            "Standard_D2s_v3",
            "westus",
            "USD",
        )
        session = MagicMock()
        session.get.return_value.__aenter__.return_value.json.return_value = {
            "Items": [
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="sku-1",
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3 Windows",  # Should be ignored
                    reservationTerm=None,
                ).dict(by_alias=True),
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="sku-1 Spot",  # Should be ignored
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3",
                    reservationTerm=None,
                ).dict(by_alias=True),
                _RetailPriceRespItem(
                    retailPrice=10,
                    skuName="sku-1 Low Priority",  # Should be ignored
                    armRegionName="westus",
                    currencyCode="USD",
                    productName="Standard D2s v3",
                    reservationTerm=None,
                ).dict(by_alias=True),
            ],
            "BillingCurrency": "USD",
        }
        await client._fetch_consumption_pricing(session)
        self.assertIsNone(client._consumption_resp)

    @patch.object(_RetailPricingClient, "_fetch_discount_pricing")
    @patch.object(_RetailPricingClient, "_fetch_consumption_pricing")
    @patch.object(_RetailPricingClient, "_fetch_spot_pricing")
    async def test_fetch_pricing__no_results(self, *_):
        client = _RetailPricingClient(
            "",
            "Standard_D2s_v3",
            "westus",
            "USD",
        )
        await client.get_pricing_info(MagicMock())
        self.assertIsNone(client._consumption_resp)
        self.assertIsNone(client._spot_resp)
        self.assertIsNone(client._discounted_1yr_resp)
        self.assertIsNone(client._discounted_3yr_resp)

    @patch.object(
        _RetailPricingClient, "_fetch_discount_pricing", side_effect=Exception
    )
    @patch.object(_RetailPricingClient, "_fetch_consumption_pricing")
    @patch.object(_RetailPricingClient, "_fetch_spot_pricing")
    async def test_fetch_pricing__internal_error(self, *_):
        client = _RetailPricingClient(
            "",
            "Standard_D2s_v3",
            "westus",
            "USD",
        )
        with self.assertRaises(InternalError):
            await client.get_pricing_info(MagicMock())

    @patch.object(_RetailPricingClient, "_fetch_discount_pricing")
    @patch.object(_RetailPricingClient, "_fetch_consumption_pricing")
    @patch.object(_RetailPricingClient, "_fetch_spot_pricing")
    async def test_fetch_pricing__all_results(
        self,
        spot_mock,
        consumption_mock,
        discount_mock,
    ):
        pricing_1yr = 1000
        pricing_3yr = 3000
        pricing_consumption = 10
        pricing_spot = 5
        currency = "USD"
        sku = "sku-1"
        region = "westus"

        client = _RetailPricingClient(
            url="",
            sku=sku,
            region=region,
            currency=currency,
        )

        def _set_spot_resp(*_):
            client._spot_resp = _RetailPriceRespItem(
                retailPrice=pricing_spot,
                skuName="sku-1",
                armRegionName="westus",
                currencyCode="USD",
                productName="Standard D2s v3",
                reservationTerm=None,
            )

        def _set_consumption_resp(*_):
            client._consumption_resp = _RetailPriceRespItem(
                retailPrice=pricing_consumption,
                skuName="sku-1",
                armRegionName="westus",
                currencyCode="USD",
                productName="Standard D2s v3",
                reservationTerm=None,
            )

        def _set_discount_resps(*_):
            client._discounted_1yr_resp = _RetailPriceRespItem(
                retailPrice=pricing_1yr,
                skuName="sku-1",
                armRegionName="westus",
                currencyCode="USD",
                productName="Standard D2s v3",
                reservationTerm=None,
            )
            client._discounted_3yr_resp = _RetailPriceRespItem(
                retailPrice=pricing_3yr,
                skuName="sku-1",
                armRegionName="westus",
                currencyCode="USD",
                productName="Standard D2s v3",
                reservationTerm=None,
            )

        spot_mock.side_effect = _set_spot_resp
        consumption_mock.side_effect = _set_consumption_resp
        discount_mock.side_effect = _set_discount_resps

        pricing_info = await client.get_pricing_info(MagicMock())
        self.assertEqual(sku, pricing_info.sku)
        self.assertEqual(currency, pricing_info.currency)
        self.assertEqual(region, pricing_info.region)
        self.assertEqual(pricing_consumption, pricing_info.price_hr)
        self.assertEqual(pricing_spot, pricing_info.price_hr_spot)
        self.assertEqual(
            pricing_1yr / constants.HOURS_PER_YEAR,
            pricing_info.price_hr_1yr,
        )
        self.assertEqual(
            pricing_3yr / constants.HOURS_PER_YEAR / 3,
            pricing_info.price_hr_3yr,
        )
