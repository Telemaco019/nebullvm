from surfer.computing.models import VMPricingInfo
from surfer.computing.services import PricingService


class GCPPricingService(PricingService):
    async def get_vm_pricing(
        self,
        vm_size: str,
        region: str,
        currency: str = "USD",
        **kwargs,
    ) -> VMPricingInfo:
        raise NotImplementedError
