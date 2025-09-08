class StrategyData:
    def __init__(self, dataset_path: str, medmnist_size: int = 224, val_percentage: int = 10,
                 num_clients: int = 10, concentration: float = 0.1, dataset_iid: bool = True,
                 seed: int = 42, strategy: str = "fedavg", dataset_task: str = "multi-class",
                 dataset="cifar10", model_name: str = "resnet18", device: str = "cuda",
                 use_pretrain: bool = False, amp: bool = True,
                 use_groupnorm=True, groupnorm_channels: int = 2):
        self.dataset = dataset
        self.model_name = model_name
        self.device = device
        self.use_pretrain = use_pretrain
        self.use_groupnorm = use_groupnorm
        self.groupnorm_channels = groupnorm_channels
        self.dataset_path = dataset_path
        self.medmnist_size = medmnist_size
        self.val_percentage = val_percentage
        self.num_clients = num_clients
        self.concentration = concentration
        self.dataset_iid = dataset_iid
        self.seed = seed
        self.strategy = strategy
        self.dataset_task = dataset_task
        self.amp = amp
