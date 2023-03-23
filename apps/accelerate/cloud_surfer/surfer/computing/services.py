import abc

from surfer.computing import schemas
from surfer.computing.models import VMPricingInfo, VMProvider


class PricingService(abc.ABC):
    @classmethod
    def from_provider(cls, provider: VMProvider, **kwargs) -> "PricingService":
        if provider is VMProvider.AWS:
            from surfer.computing.providers import aws

            return aws.AWSPricingService()
        if provider is VMProvider.GCP:
            from surfer.computing.providers import gcp

            return gcp.GCPPricingService()
        if provider is VMProvider.AZURE:
            from surfer.computing.providers import azure

            return azure.AzurePricingService(**kwargs)

        raise ValueError(f"provider {provider} is not supported")

    @abc.abstractmethod
    async def get_vm_pricing(
        self,
        vm_sku: str,
        region: str,
        currency: str = "USD",
        **kwargs,
    ) -> schemas.VMPricing:
        """
        Get pricing information for a given VM size in a certain region.

        Parameters
        ----------
        vm_sku: str
            The VM size (e.g. the SKU) to get pricing information for.
            The format of the VM size depends on the specific cloud provider.
            Examples:
                Azure:
                    - Standard_D2s_v3
                    - Standard_NC6s_v3
                GCP:
                    - n1-standard-4
        region: str
            The region to get pricing information for. The format
            of the region depends on the specific cloud provider
            (e.g. Azure, AWS, GCP).
            Examples:
                Azure:
                    - westeurope
                    - eastus
                GCP:
                    - us-central1
        currency: str
            The currency to get the pricing information in.
            Defaults to USD.

        Returns
        -------
        VMPricingInfo
            The pricing information for the given VM size in the given region.
        """
