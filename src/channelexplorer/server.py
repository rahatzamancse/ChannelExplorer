from abc import ABC, abstractmethod
from typing import Literal
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pathlib
from fastapi.middleware.cors import CORSMiddleware

# Create abstract class
class Server(ABC):
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
        """
        __init__ Creates a ChannelExplorer object.

        :param model: The TensorFlow model to be analyzed.
        :type model: K.Model
        :param dataset: The dataset used to analyze the model.
        :type dataset: tf.data.Dataset
        :param label_names: The label mapping in an array. ith element is the label of class i, defaults to []
        :type label_names: list[str], optional
        :param summary_fn_image: The summarization function to use to convert an activation function to a real value, defaults to metrics.summary_fn_image_l2
        :type summary_fn_image: Callable[ [IMAGE_BATCH_TYPE], SUMMARY_BATCH_TYPE ], optional
        :param summary_fn_dense: The summarization function to use to convert the activation of a neuron in Dense layer to real value, defaults to metrics.summary_fn_dense_identity
        :type summary_fn_dense: Callable[ [DENSE_BATCH_TYPE], SUMMARY_BATCH_TYPE ], optional
        :param log_level: The logging level, defaults to "info"
        :type log_level: Literal[&quot;info&quot;, &quot;debug&quot;], optional
        :param layers_to_show: Names of the layers to visualize in the frontend, defaults to "all"
        :type layers_to_show: list[str] | Literal[&quot;all&quot;], optional
        :return: The server object.
        :rtype: Server
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
        """
        run the server

        :param host: host ip, defaults to "0.0.0.0"
        :type host: str, optional
        :param port: port, defaults to 8000
        :type port: int, optional
        """
        # Starting the server
        # self.app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parents[0].joinpath('static').resolve(), html=True), name="react_build")
        uvicorn.run(self.app, host=host, port=port, log_level=self.log_level)
