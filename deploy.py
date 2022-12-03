from sieve.api.client import SieveClient, SieveProject
from sieve.types.api import *
cli = SieveClient()

proj = SieveProject(
    name="custom_emotion_detection2",
    fps=5,
    store_data=True,
    workflow=SieveWorkflow([
        SieveLayer(
            iteration_type=SieveLayerIterationType.video,
            models=[
                SieveModel(
                    name="developer-sievedata-com/face-detector",
                )
            ]
        ),
        SieveLayer(
            iteration_type=SieveLayerIterationType.objects,
            models=[
                SieveModel(
                    name="emotion-detector2"
                )
            ]
        )
    ])
)
proj.create()