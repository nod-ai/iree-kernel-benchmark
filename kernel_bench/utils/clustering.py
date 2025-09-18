import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from typing import List, Tuple, Dict, Union, Optional, Any
import pandas as pd


class KernelConfigurationClustering:
    """
    A clustering algorithm for finding representative samples of kernel configurations.

    This class can handle kernel configurations with both numerical and categorical features.
    """

    def __init__(
        self,
        feature_types: Optional[List[str]] = None,
        categorical_encoding: str = "onehot",
        scaling_method: str = "standard",
        clustering_method: str = "kmeans",
        n_clusters: Optional[int] = None,
        min_clusters: int = 2,
        max_clusters: int = 20,
        random_state: int = 42,
    ):
        """
        Initialize the kernel configuration clustering algorithm.

        Args:
            feature_types: List specifying type of each feature ('numerical' or 'categorical').
                          If None, will try to infer from data.
            categorical_encoding: How to encode categorical variables ('onehot' or 'label')
            scaling_method: Method for scaling numerical features ('standard', 'minmax', or 'none')
            clustering_method: Clustering algorithm to use ('kmeans', 'dbscan', or 'hierarchical')
            n_clusters: Fixed number of clusters (if None, will be determined automatically)
            min_clusters: Minimum number of clusters to try (for automatic determination)
            max_clusters: Maximum number of clusters to try (for automatic determination)
            random_state: Random seed for reproducibility
        """
        self.feature_types = feature_types
        self.categorical_encoding = categorical_encoding
        self.scaling_method = scaling_method
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state

        self.preprocessor = None
        self.clusterer = None
        self.configurations = None
        self.configurations_df = None
        self.transformed_configurations = None
        self.labels = None
        self.representatives = None
        self.label_encoders = {}

    def _infer_feature_types(self, data: pd.DataFrame) -> List[str]:
        """Infer whether each feature is numerical or categorical."""
        feature_types = []

        for col in data.columns:
            # Check if column contains strings or has few unique values relative to total
            if data[col].dtype == "object" or (
                data[col].nunique() < 10 and data[col].nunique() / len(data) < 0.05
            ):
                feature_types.append("categorical")
            else:
                feature_types.append("numerical")

        return feature_types

    def _create_preprocessor(self, data: pd.DataFrame) -> ColumnTransformer:
        """Create a preprocessor for handling mixed data types."""
        if self.feature_types is None:
            self.feature_types = self._infer_feature_types(data)

        numerical_features = []
        categorical_features = []

        for i, (col, ftype) in enumerate(zip(data.columns, self.feature_types)):
            if ftype == "numerical":
                numerical_features.append(col)
            else:
                categorical_features.append(col)

        transformers = []

        # Add numerical transformer if needed
        if numerical_features:
            if self.scaling_method == "standard":
                numerical_transformer = StandardScaler()
            elif self.scaling_method == "minmax":
                numerical_transformer = MinMaxScaler()
            else:
                numerical_transformer = "passthrough"

            transformers.append(("num", numerical_transformer, numerical_features))

        # Add categorical transformer if needed
        if categorical_features:
            if self.categorical_encoding == "onehot":
                categorical_transformer = OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore"
                )
            else:
                # For label encoding, we'll handle it separately to maintain mappings
                categorical_transformer = "passthrough"

            transformers.append(("cat", categorical_transformer, categorical_features))

        return ColumnTransformer(transformers=transformers)

    def _preprocess_data(self, data: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Preprocess the data with appropriate transformations."""
        if fit:
            # If using label encoding for categorical variables, handle separately
            if self.categorical_encoding == "label":
                data_encoded = data.copy()

                for i, (col, ftype) in enumerate(zip(data.columns, self.feature_types)):
                    if ftype == "categorical":
                        le = LabelEncoder()
                        data_encoded[col] = le.fit_transform(data[col].astype(str))
                        self.label_encoders[col] = le

                # Create preprocessor for the encoded data
                self.preprocessor = self._create_preprocessor(data_encoded)
                return self.preprocessor.fit_transform(data_encoded)
            else:
                self.preprocessor = self._create_preprocessor(data)
                return self.preprocessor.fit_transform(data)
        else:
            # Transform only (for new data)
            if self.categorical_encoding == "label":
                data_encoded = data.copy()

                for col, le in self.label_encoders.items():
                    if col in data.columns:
                        # Handle unknown categories
                        known_classes = set(le.classes_)
                        data_encoded[col] = data[col].apply(
                            lambda x: (
                                le.transform([str(x)])[0]
                                if str(x) in known_classes
                                else -1
                            )
                        )

                return self.preprocessor.transform(data_encoded)
            else:
                return self.preprocessor.transform(data)

    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """Find the optimal number of clusters using silhouette score."""
        if len(data) < self.min_clusters:
            return 1

        max_clusters = min(self.max_clusters, len(data))
        best_score = -1
        best_n_clusters = self.min_clusters

        for n in range(self.min_clusters, max_clusters + 1):
            if n >= len(data):
                break

            kmeans = KMeans(n_clusters=n, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(data)

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(data, labels)

            if score > best_score:
                best_score = score
                best_n_clusters = n

        return best_n_clusters

    def _get_representative(self, cluster_indices: List[int]) -> int:
        """Get the representative configuration for a cluster."""
        cluster_data = self.transformed_configurations[cluster_indices]
        centroid = np.mean(cluster_data, axis=0)
        distances = np.linalg.norm(cluster_data - centroid, axis=1)
        return cluster_indices[np.argmin(distances)]

    def fit(
        self, configurations: Union[List[Tuple], List[List], pd.DataFrame]
    ) -> "KernelConfigurationClustering":
        """
        Fit the clustering algorithm to the kernel configurations.

        Args:
            configurations: Can be:
                - List of tuples/lists where each element is a kernel configuration
                - DataFrame where each row is a kernel configuration

        Returns:
            self: The fitted clustering object
        """
        # Convert to DataFrame for easier handling
        if isinstance(configurations, pd.DataFrame):
            self.configurations_df = configurations.copy()
        else:
            self.configurations_df = pd.DataFrame(configurations)
            # Generate column names if not provided
            self.configurations_df.columns = [
                f"feature_{i}" for i in range(len(configurations[0]))
            ]

        # Store original configurations
        self.configurations = configurations

        # Preprocess the data
        self.transformed_configurations = self._preprocess_data(
            self.configurations_df, fit=True
        )

        # Determine number of clusters if not specified
        if self.n_clusters is None:
            self.n_clusters = self._find_optimal_clusters(
                self.transformed_configurations
            )

        # Apply clustering algorithm
        if self.clustering_method == "kmeans":
            self.clusterer = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
            )
        elif self.clustering_method == "dbscan":
            from sklearn.neighbors import NearestNeighbors

            neighbors = NearestNeighbors(
                n_neighbors=min(5, len(self.transformed_configurations))
            )
            neighbors.fit(self.transformed_configurations)
            distances, _ = neighbors.kneighbors(self.transformed_configurations)
            distances = np.sort(distances[:, -1])
            eps = np.percentile(distances, 90)

            self.clusterer = DBSCAN(eps=eps, min_samples=2)
        elif self.clustering_method == "hierarchical":
            self.clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        # Fit and predict labels
        self.labels = self.clusterer.fit_predict(self.transformed_configurations)

        # Handle noise points in DBSCAN
        if self.clustering_method == "dbscan" and -1 in self.labels:
            noise_indices = np.where(self.labels == -1)[0]
            max_label = self.labels.max()
            for i, idx in enumerate(noise_indices):
                self.labels[idx] = max_label + i + 1

        return self

    def get_representatives(self) -> List[Union[Tuple, List]]:
        """Get the representative configurations from each cluster."""
        if self.labels is None:
            raise ValueError("Model must be fitted before getting representatives")

        self.representatives = []
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            cluster_indices = np.where(self.labels == label)[0].tolist()
            representative_idx = self._get_representative(cluster_indices)

            # Return in the same format as input
            if isinstance(self.configurations, pd.DataFrame):
                rep = self.configurations.iloc[representative_idx].tolist()
            elif isinstance(self.configurations, list) and isinstance(
                self.configurations[0], tuple
            ):
                rep = tuple(self.configurations_df.iloc[representative_idx].tolist())
            else:
                rep = self.configurations_df.iloc[representative_idx].tolist()

            self.representatives.append(rep)

        return self.representatives

    def get_cluster_info(self) -> pd.DataFrame:
        """Get detailed information about each cluster."""
        if self.labels is None:
            raise ValueError("Model must be fitted before getting cluster info")

        cluster_info = []
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            cluster_indices = np.where(self.labels == label)[0]
            cluster_df = self.configurations_df.iloc[cluster_indices]

            info = {
                "cluster_id": label,
                "size": len(cluster_indices),
            }

            # Add representative
            rep_idx = self._get_representative(cluster_indices.tolist())
            rep_config = self.configurations_df.iloc[rep_idx]
            for col in self.configurations_df.columns:
                info[f"representative_{col}"] = rep_config[col]

            # Add statistics for numerical features
            for i, (col, ftype) in enumerate(
                zip(self.configurations_df.columns, self.feature_types)
            ):
                if ftype == "numerical":
                    info[f"{col}_mean"] = cluster_df[col].mean()
                    info[f"{col}_std"] = cluster_df[col].std()
                    info[f"{col}_min"] = cluster_df[col].min()
                    info[f"{col}_max"] = cluster_df[col].max()
                else:
                    # For categorical, show mode and distribution
                    mode_val = (
                        cluster_df[col].mode().iloc[0]
                        if not cluster_df[col].mode().empty
                        else None
                    )
                    info[f"{col}_mode"] = mode_val
                    info[f"{col}_unique_count"] = cluster_df[col].nunique()

            cluster_info.append(info)

        return pd.DataFrame(cluster_info)

    def predict_cluster(
        self, new_configurations: Union[List[Tuple], List[List], pd.DataFrame]
    ) -> np.ndarray:
        """Predict which cluster new configurations belong to."""
        if self.clusterer is None:
            raise ValueError("Model must be fitted before predicting")

        # Convert to DataFrame
        if isinstance(new_configurations, pd.DataFrame):
            new_df = new_configurations.copy()
        else:
            new_df = pd.DataFrame(
                new_configurations, columns=self.configurations_df.columns
            )

        # Preprocess the new data
        transformed_new = self._preprocess_data(new_df, fit=False)

        # Predict clusters
        if hasattr(self.clusterer, "predict"):
            return self.clusterer.predict(transformed_new)
        else:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(self.transformed_configurations)
            _, indices = nn.kneighbors(transformed_new)
            return self.labels[indices.flatten()]
