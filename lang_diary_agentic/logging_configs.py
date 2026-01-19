import tqdm
import os
import logging

from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Specifically for the XLA/absl warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'


def apply_logging_suppressions():

    # ---- Monkey-patch tqdm to disable it globally ----
    # I want to disable tqdm shown in the Mlflow
    tqdm.__init__ = partial(tqdm.__init__, disable=True)
    # ---- 

    # Silence the 'urllib3' library (this stops the connectionpool messages)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Silence Hugging Face 'transformers' and 'datasets' specifically
    # Hugging Face provides their own helper to set verbosity
    try:
        import transformers
        transformers.utils.logging.set_verbosity_warning()
    except ImportError:
        pass

    try:
        import datasets
        datasets.utils.logging.set_verbosity_warning()
    except ImportError:
        pass


    # 3. In some environments, 'requests' also has its own logger
    logging.getLogger("requests").setLevel(logging.WARNING)
