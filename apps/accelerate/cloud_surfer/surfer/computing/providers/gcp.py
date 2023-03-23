from surfer.computing import schemas
from surfer.computing.services import PricingService


class GCPPricingService(PricingService):
    async def get_vm_pricing(
        self,
        vm_size: str,
        region: str,
        currency: str = "USD",
        **kwargs,
    ) -> schemas.VMPricing:
        raise NotImplementedError
