from surfer.computing import schemas
from surfer.computing.services import PricingService


class AWSPricingService(PricingService):
    async def get_vm_pricing(
        self,
        vm_sku: str,
        region: str,
        currency: str = "USD",
        **kwargs,
    ) -> schemas.VMPricing:
        raise NotImplementedError
