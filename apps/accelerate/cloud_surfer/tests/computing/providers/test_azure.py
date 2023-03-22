import unittest
from unittest.mock import patch, AsyncMock

import aiohttp

from surfer.computing.providers.azure import AzurePricingService


class TestAzurePricingService(unittest.IsolatedAsyncioTestCase):
    @patch.object(aiohttp.ClientSession, "get")
    async def test_get_vm_pricing__stop_not_found(self, get_mock):
        service = AzurePricingService()
        get_mock.return_value.__aenter__.return_value.json = AsyncMock(
            return_value={}
        )
        await service.get_vm_pricing(
            vm_size="Standard_D2s_v3",
            region="westus",
            currency="USD",
        )
