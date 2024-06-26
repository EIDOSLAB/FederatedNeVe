from NeVe.federated.flwr.strategies.sampler.aclient_sampler import ClientSampler
from NeVe.federated.flwr.strategies.sampler.default_sampler import DefaultSampler
from NeVe.federated.flwr.strategies.sampler.logger.client_logger import ClientSamplerLogger
from NeVe.federated.flwr.strategies.sampler.percentage_random_sampler import PercentageRandomSampler
from NeVe.federated.flwr.strategies.sampler.percentage_rgroups_sampler import PercentageRGroupsSampler
from NeVe.federated.flwr.strategies.sampler.velocity_sampler import VelocitySampler


def get_client_sampler(sampling_method: str, sampling_percentage: float = 0.5, sampling_wait_epochs: int = 10,
                       sampling_velocity_aging: float = 0.01, sampling_highest_velocity: bool = True,
                       sampling_min_epochs: int = 2, sampling_use_probability: bool = True) -> ClientSampler:
    logger = ClientSamplerLogger()
    match (sampling_method.lower()):
        case "default":
            sampler = DefaultSampler(logger)
        case "percentage_random":
            sampler = PercentageRandomSampler(logger, sampling_wait_epochs=sampling_wait_epochs,
                                              clients_sampling_percentage=sampling_percentage)
        case "percentage_groups":
            sampler = PercentageRGroupsSampler(logger, sampling_wait_epochs=sampling_wait_epochs,
                                               clients_sampling_percentage=sampling_percentage)
        case "velocity":
            sampler = VelocitySampler(logger, sampling_wait_epochs=sampling_wait_epochs,
                                      clients_sampling_percentage=sampling_percentage,
                                      sampling_velocity_aging=sampling_velocity_aging,
                                      sampling_highest_velocity=sampling_highest_velocity,
                                      sampling_min_epochs=sampling_min_epochs,
                                      sampling_use_probability=sampling_use_probability)
        case _:
            raise Exception(f"Client Sampler '{sampling_method.lower()}' not defined!")
    return sampler
