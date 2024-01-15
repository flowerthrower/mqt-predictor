from __future__ import annotations


# class GNNClassifier:
#    def __init__(
#        self,x
#        optimizer: str = "adam",
#        learning_rate: float = 1e-3,
#        batch_size: int = 64,
#        epochs: int = 10,
#        num_node_categories=42,  # distinct gate types (incl. 'id' and 'meas')
#        num_edge_categories=100,  # distinct wires (quantum + classical)
#        node_embedding_dim=4,  # dimension of the node embedding
#        edge_embedding_dim=4,  # dimension of the edge embedding
#        num_layers=2,  # number of nei aggregations
#        hidden_dim=16,  # dimension of the hidden layers
#        output_dim=3,  # dimension of the output vector
#    ):
#        self.set_params(
#            optimizer=optimizer,
#            learning_rate=learning_rate,
#            batch_size=batch_size,
#            epochs=epochs,
#            num_node_categories=num_node_categories,
#            num_edge_categories=num_edge_categories,
#            node_embedding_dim=node_embedding_dim,
#            edge_embedding_dim=edge_embedding_dim,
#            num_layers=num_layers,
#            hidden_dim=hidden_dim,
#            output_dim=output_dim,
#        )
#        # Initialize the model
#        self.model = Net(
#            num_node_categories,
#            num_edge_categories,
#            node_embedding_dim,
#            edge_embedding_dim,
#            num_layers,
#            hidden_dim,
#            output_dim,
#        )
#
#        if optimizer == "adam":
#            self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
#        elif optimizer == "sgd":
#            self.optim = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
#
#    def fit(self, X, y=None):
#        self.model.train()
#        loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)
#
#        for _ in range(self.epochs):
#            for batch in loader:
#                self.optim.zero_grad()
#                out = self.model.forward(batch)
#                loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
#                loss.backward()
#                self.optim.step()
#
#    def predict(self, data):
#        self.model.eval()
#        logits = self.model.forward(data)
#        return logits.argmax(dim=-1)
#
#    def score(self, X, y):
#        y_pred = self.predict(X)
#        return accuracy_score(y, y_pred)
#
#    def get_params(self):
#        return {
#            "optimizer": self.optimizer,
#            "learning_rate": self.learning_rate,
#            "batch_size": self.batch_size,
#            "epochs": self.epochs,
#            "num_node_categories": self.num_node_categories,
#            "num_edge_categories": self.num_edge_categories,
#            "node_embedding_dim": self.node_embedding_dim,
#            "edge_embedding_dim": self.edge_embedding_dim,
#            "num_layers": self.num_layers,
#            "hidden_dim": self.hidden_dim,
#            "output_dim": self.output_dim,
#        }
#
#    def set_params(self, **params: dict):
#        for parameter, value in params.items():
#            setattr(self, parameter, value)
#        return self
#
