import logging
from dataclasses import dataclass
from typing import Optional

import torch
from cpufeature import CPUFeature
from petals.constants import PUBLIC_INITIAL_PEERS


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# Model configuration
# ============================================================

@dataclass
class ModelInfo:
    repo: str
    adapter: Optional[str] = None


MODELS = []


# ============================================================
# Petals configuration
# ============================================================

# Use the public Petals swarm.
INITIAL_PEERS = PUBLIC_INITIAL_PEERS


# ============================================================
# Hardware configuration
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    TORCH_DTYPE = torch.float16

    logger.info(
        "CUDA available: %s",
        torch.cuda.get_device_name(0),
    )

else:
    DEVICE = torch.device("cpu")

    try:
        cpu_features = CPUFeature["auto"]
        os_features = CPUFeature["OS_auto"]

        if cpu_features and os_features:
            TORCH_DTYPE = torch.bfloat16
        else:
            TORCH_DTYPE = torch.float32

    except Exception as exc:
        logger.warning(
            "CPU feature detection failed: %s. "
            "Falling back to float32.",
            exc,
        )
        TORCH_DTYPE = torch.float32


# ============================================================
# Runtime configuration
# ============================================================

STEP_TIMEOUT = 10 * 60
MAX_SESSIONS = 50

logger.info("Configuration setup complete.")
logger.info("Device: %s", DEVICE)
logger.info("Torch dtype: %s", TORCH_DTYPE)
logger.info("Step timeout: %s seconds", STEP_TIMEOUT)
logger.info("Max sessions: %s", MAX_SESSIONS)


# ============================================================
# Preprocessing
# ============================================================

def preprocess(data: torch.Tensor) -> torch.Tensor:
    logger.debug("Preprocessing data")
    return data


# ============================================================
# Postprocessing
# ============================================================

def postprocess(data: torch.Tensor) -> torch.Tensor:
    logger.debug("Postprocessing data")
    return data


# ============================================================
# Example model
# ============================================================

class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# Initialize model
# ============================================================

model = MyModel().to(
    device=DEVICE,
    dtype=TORCH_DTYPE,
)

model.eval()


# ============================================================
# Hybrid CPU/GPU inference
# ============================================================

@torch.inference_mode()
def hybrid_function(data: torch.Tensor) -> torch.Tensor:

    logger.debug("Starting hybrid inference")

    # CPU preprocessing
    data_cpu = data.to(
        device="cpu",
        dtype=torch.float32,
    )

    preprocessed_data = preprocess(data_cpu)

    # GPU/CPU inference
    preprocessed_data = preprocessed_data.to(
        device=DEVICE,
        dtype=TORCH_DTYPE,
    )

    output = model(preprocessed_data)

    # CPU postprocessing
    output_cpu = output.to(
        device="cpu",
        dtype=torch.float32,
    )

    result = postprocess(output_cpu)

    logger.debug("Hybrid inference complete")

    return result


# ============================================================
# Example execution
# ============================================================

if __name__ == "__main__":

    logger.info("Starting application")

    data = torch.randn(
        100,
        10,
        device="cpu",
        dtype=torch.float32,
    )

    result = hybrid_function(data)

    logger.info(
        "Processing complete. "
        "Input shape=%s, output shape=%s",
        tuple(data.shape),
        tuple(result.shape),
    )

    print("Device:", DEVICE)
    print("Dtype:", TORCH_DTYPE)
    print("Result shape:", result.shape)
