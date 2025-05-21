import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

RANDOM_SEED_MAX = 255


def seed_everything(seed: Optional[int] = None) -> int:
    """Set random seed for reproducibility."""
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = random.randint(0, RANDOM_SEED_MAX)
            logger.warning(f"No seed found, seed set to {seed}")  # noqa: G004
        else:
            seed = int(env_seed)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)  # noqa: NPY002
    logger.info(f"Global seed set to {seed}")  # noqa: G004

    return seed
