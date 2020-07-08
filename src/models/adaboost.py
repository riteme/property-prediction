from typing import Optional, Type, Text, IO

import sys

import log
from .base import *
from .gat import GAT
from .graphsage import GraphSAGE

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier


def select_model(inner_model: Text) -> Type[EmbeddableModel]:
    if inner_model == 'gat':
        return GAT
    elif 'sage' in inner_model:
        return GraphSAGE
    else:
        log.fatal(f'Invalid "inner_model": "{inner_model}"')

class AdaBoost(BaseModel):
    model_type: Type[EmbeddableModel]

    def __init__(self,
        device: torch.device,
        init_model: Optional[IO] = None,
        num_estimator: int = 100,
        hard_predict: bool = False,
        inner_model: Text = 'gat',
        **kwargs
    ):
        super().__init__(device)

        self.model_type = select_model(inner_model)
        self._inst = self.model_type(device, **kwargs)
        if init_model is not None:
            state_dict = torch.load(init_model)
            self._inst.load_state_dict(state_dict)

        self.classifier = Pipeline([
            ('scalar', StandardScaler()),
            ('adaboost', CalibratedClassifierCV(AdaBoostClassifier(n_estimators=num_estimator)))
        ])

        self.hard_predict = hard_predict

    @staticmethod
    def decode_data(
        data: Any, device: torch.device,
        inner_model: Text = 'gat',
        **kwargs
    ) -> Any:
        model_type = select_model(inner_model)
        return model_type.decode_data(data, device)

    @staticmethod
    def process(
        mol: Mol, device: torch.device,
        inner_model: Text = 'gat',
        **kwargs
    ) -> Any:
        model_type = select_model(inner_model)
        return model_type.process(mol, device, **kwargs)

    def forward(self, data):
        return self._inst.forward(data)

    def postprocess(self, train_data: List[Item]):
        X = [
            self._inst.embed(x.obj).tolist()
            for x in train_data
        ]
        y = [x.activity for x in train_data]

        # weight distribution:
        #   for positive samples: #zero
        #   for negative samples: #one
        #   sum of weights: #one * #zero + #zero * #one → denom
        one_count = float(sum(y))
        zero_count = float(len(y) - one_count)
        w = [
            zero_count if label == 1 else one_count
            for label in y
        ]
        denom = sum(w)
        w = [x / denom for x in w]

        self.classifier.fit(X, y, adaboost__sample_weight=w)

    def predict(self, data: Any) -> torch.Tensor:
        if self.in_training:
            return self._inst.predict(data)
        else:
            X = self._inst.embed(data).cpu().numpy()[None, :]
            if self.hard_predict:
                who = self.classifier.predict(X)[0]
                return torch.tensor([1 - who, who])
            else:
                result = self.classifier.predict_proba(X)[0]
                return torch.tensor(result)
