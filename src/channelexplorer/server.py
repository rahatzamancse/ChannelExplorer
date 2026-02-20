from abc import ABC, abstractmethod
from typing import Literal
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pathlib
from fastapi.middleware.cors import CORSMiddleware

# Create abstract class
class Server(ABC):
    """Abstract base server for ChannelExplorer backends. """

    def __init__(
            self,
            model,
            dataset,
            label_names,
            summary_fn_image,
            summary_fn_dense,
            log_level: str = "info",
            layers_to_show: list[str] | Literal["all"] = "all",
        ) -> None:
        """Initialize the server with a model, dataset, and configuration.

        Args:
            model: The neural network model to analyze.
            dataset: The dataset used to analyze the model.
            label_names: Human-readable label names where the i-th element
                is the name of class *i*.
            summary_fn_image: Function that reduces a spatial activation map
                to a scalar summary per channel.
            summary_fn_dense: Function that reduces dense-layer activations
                to a scalar summary.
            log_level: Uvicorn logging level. ``"info"`` or ``"debug"``.
            layers_to_show: Layer names to expose in the frontend, or
                ``"all"`` to include every supported layer.
        """
        super().__init__()
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.log_level = log_level


    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """Start the FastAPI server with Uvicorn.

        Args:
            host: IP address to bind to.
            port: Port number to listen on.
        """
        static_dir = pathlib.Path(__file__).parent / "static"
        if static_dir.is_dir():
            self.app.mount("/", StaticFiles(directory=static_dir, html=True), name="static_frontend")
        uvicorn.run(self.app, host=host, port=port, log_level=self.log_level)
