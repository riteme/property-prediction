from .base import *

from .gat import GAT, GATData

import log

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class SVM(BaseModel):
    def __init__(self,
        device: torch.device,
        **kwargs
    ):
        super().__init__(device)

        self._inst = GAT(device)
        self.classifier = make_pipeline(
            StandardScaler(),
            SVC(
                probability=True,
                kernel='rbf',
                cache_size=512,
                class_weight='balanced'
            )
        )

    @staticmethod
    def decode_data(data: GATData, device: torch.device, **kwargs) -> GATData:
        return GAT.decode_data(data, device)

    @staticmethod
    def process(mol: Mol, device: torch.device, **kwargs) -> GATData:
        return GAT.process(mol, device)

    def forward(self, data):
        return self._inst.forward(data)

    def postprocess(self, train_data: List[Item]):
        X = [
            self._inst.embed(x.obj).tolist()
            for x in train_data
        ]
        y = [x.activity for x in train_data]

        self.classifier.fit(X, y)

    def predict(self, data: Any) -> torch.Tensor:
        if self.in_training:
            return self._inst.predict(data)
        else:
            X = self._inst.embed(data).cpu().numpy()[None, :]
            pred = self.classifier.predict_proba(X)[0]
            log.debug(pred)
            return torch.tensor(pred)
            # who = self.classifier.predict(X)[0]
            # return torch.tensor([1 - who, who])
