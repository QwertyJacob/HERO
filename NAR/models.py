import torch
import torch.nn as nn


def get_kernel(
        encoded_inputs,
        masked_oh_labels,
        sparsity,
        device='cpu'):
    """
    The known region has a semantic kernel, the unknown region has a 
    KNN euclidean kernel, and the known-unknown  and unknown-known relationships
    are initialized as in the paper of Bruna (i.e. fully connected but normalized)
    """
    # get number of elements
    n_elements = encoded_inputs.shape[0]
    # kernel initialization.
    kernel = torch.zeros(
        [n_elements, n_elements],
        device=device)

    # unknown mask
    unknown_mask = masked_oh_labels.sum(1) == 0
    # known mask:
    known_mask = ~unknown_mask

    # Semantic Kernel:
    # we need a tensors indicating the indexes to fill the kernel
    known_indexes_tensor = known_mask\
        .unsqueeze(0)\
        .repeat(n_elements, 1)
    # mask the unknown rows to False
    known_indexes_tensor[unknown_mask] = False

    # semantic kernel:
    semantic_kernel = masked_oh_labels @ masked_oh_labels.T

    # fill the known region of the kernel with the semantic kernel:
    kernel = torch.where(
        condition=known_indexes_tensor,
        input=semantic_kernel,
        other=kernel)

    # KNN Kernel:
    if torch.any(unknown_mask):
        # we need tensors indicating the indexes to fill the kernel
        unknown_indexes_tensor = unknown_mask\
            .unsqueeze(0)\
            .repeat(n_elements, 1)
        # mask the whole known rows to False
        unknown_indexes_tensor[known_mask] = False

        # distances between points:
        euclidean_distances = torch.cdist(
            encoded_inputs, encoded_inputs)
        # symmetrize:
        euclidean_distances = (
            euclidean_distances + euclidean_distances.t()) / 2
        # compute adagae kernel:
        adagae_kernel = adagae_regularization(
            euclidean_distances, sparsity)
        # fill the unknown-unknown region with the adagae kernel:
        kernel = torch.where(
            condition=unknown_indexes_tensor,
            input=adagae_kernel,
            other=kernel)

        # Fully-connected Kernel:
        fc_kernel = torch.ones(
            [n_elements, n_elements],
            device=device)
        # normalize:
        fc_kernel = fc_kernel / torch.sum(unknown_mask)
        # heterogeneous region mask:
        heterogeneous_indexes_tensor = ~torch.logical_or(
            known_indexes_tensor,
            unknown_indexes_tensor)
        # fill the heterogeneous region with the fc kernel:
        kernel = torch.where(
            condition=heterogeneous_indexes_tensor,
            input=fc_kernel,
            other=kernel)
        assert torch.all(kernel[known_mask][:, known_mask]
                         == semantic_kernel[known_mask][:, known_mask])
        assert torch.all(kernel[unknown_mask][:, unknown_mask]
                         == adagae_kernel[unknown_mask][:, unknown_mask])
        assert torch.all(kernel[unknown_mask][:, known_mask]
                         == fc_kernel[unknown_mask][:, known_mask])
        assert torch.all(kernel[known_mask][:, unknown_mask]
                         == fc_kernel[known_mask][:, unknown_mask])
    return kernel


def adagae_regularization(distances, k):

    n = distances.shape[0]
    sorted_distances, _ = distances.sort(dim=1)

    # distance to the k-th nearest neighbour for each element:
    a = sorted_distances[:, k]
    # we boradcast that distance n times
    a = a.repeat(n, 1)
    # Horizontal broadcast is what we need (default is vertical broadcast)
    a = a.T + 10 ** -10

    # k-nearest distances per element:
    b = sorted_distances[:, :k]
    # we sum those distances:
    b = torch.sum(b, dim=1)
    # broadcast:
    b = b.repeat(n, 1)
    # vertical
    b = b.T

    # For each element, we subtract the value of all the distances to the
    # broadcasted distance to the k-th nearest neighbour.
    # Only k- nearest neighbours will get a positive number.
    c = a - distances
    # regularization:
    c = c.relu()

    # for each element, we compute the product between the sparsity k
    # and  distance to the k-th nearest neighbour for each element
    a = a * k
    # we subtrack the sum of the k-nearest distances to that quantity:
    a = a - b
    # we add a non-zero value (this is going to be at the denominator)
    a += 1e-10
    # we divide the regularized distances by this quantity:
    a = c / a
    # symmetrize
    a = (a + a.t()) / 2

    return a


def knn_regularization(distances, k):
    n = distances.shape[0]
    sorted_distances, _ = distances.sort(dim=1)

    # distance to the k-th nearest neighbour for each element:
    a = sorted_distances[:, k]
    # we boradcast that distance n times
    a = a.repeat(n, 1)
    # Horizontal broadcast is what we need (default is vertical broadcast)
    a = a.T + 10 ** -10

    # For each element, we subtract the value of all the distances to the
    # broadcasted distance to the k-th nearest neighbour.
    # Only k- nearest neighbours will get a positive number.
    c = a - distances
    # regularization:
    c = c.relu()
    # scaling:
    c = c / (1.5 * c.max(1)[0])
    return c


def get_knn_kernel(
        encoded_inputs,
        sparsity,
        device='cpu'):
    
    # distances between points:
    euclidean_distances = torch.cdist(
        encoded_inputs, encoded_inputs)

    # symmetrize:
    euclidean_distances = (
        euclidean_distances + euclidean_distances.t()) / 2

    # compute KNN kernel:
    KNN_kernel = knn_regularization(
        euclidean_distances, sparsity)

    return KNN_kernel


def get_kernel_v2(
        encoded_inputs,
        masked_oh_labels,
        sparsity,
        device='cpu'):
    """
    The known region has a semantic kernel, the unknown
    and the known-unknown  and unknown-known relationships
    are region have a KNN kernel
    """
    # get number of elements
    n_elements = encoded_inputs.shape[0]
    # kernel initialization.
    kernel = torch.zeros(
        [n_elements, n_elements],
        device=device)

    # unknown mask
    unknown_mask = masked_oh_labels.sum(1) == 0
    # known mask:
    known_mask = ~unknown_mask
    # we need a tensors indicating the indexes to fill the kernel
    known_indexes_tensor = known_mask\
        .unsqueeze(0)\
        .repeat(n_elements, 1)
    # mask the unknown rows to False
    known_indexes_tensor[unknown_mask] = False

    # semantic kernel:
    semantic_kernel = masked_oh_labels @ masked_oh_labels.T

    # fill the known region of the kernel with the semantic kernel:
    kernel = torch.where(
        condition=known_indexes_tensor,
        input=semantic_kernel,
        other=kernel)

    # KNN kernel:
    if torch.any(unknown_mask):
        # we need tensors indicating the indexes to fill the kernel
        unknown_indexes_tensor = unknown_mask\
            .unsqueeze(0)\
            .repeat(n_elements, 1)
        # mask the whole unknown rows to True (known-unknown connectivities)
        unknown_indexes_tensor[unknown_mask] = True

        # distances between points:
        euclidean_distances = torch.cdist(
            encoded_inputs, encoded_inputs)
        # symmetrize:
        euclidean_distances = (
            euclidean_distances + euclidean_distances.t()) / 2
        # compute KNN kernel:
        KNN_kernel = knn_regularization(
            euclidean_distances, sparsity)
        # fill the unknown elements with a KNN kernel:
        kernel = torch.where(
            condition=unknown_indexes_tensor,
            input=KNN_kernel,
            other=kernel)

    return kernel

def get_smart_kernel(
        encoded_inputs,
        masked_oh_labels,
        sparsity,
        device='cpu'):
    
    # get number of elements
    n_elements = encoded_inputs.shape[0]
    # kernel initialization.
    kernel = torch.zeros(
        [n_elements, n_elements],
        device=device)

    # unknown mask
    unknown_mask = masked_oh_labels.sum(1) == 0
    # known mask:
    known_mask = ~unknown_mask

    # Semantic Kernel:
    # we need a tensors indicating the indexes to fill the kernel
    known_indexes_tensor = known_mask\
        .unsqueeze(0)\
        .repeat(n_elements, 1)
    # mask the unknown rows to False
    known_indexes_tensor[unknown_mask] = False

    # semantic kernel:
    semantic_kernel = masked_oh_labels @ masked_oh_labels.T

    # fill the known region of the kernel with the semantic kernel:
    kernel = torch.where(
        condition=known_indexes_tensor,
        input=semantic_kernel,
        other=kernel)

    # KNN Kernel: ONLY IN THE MIXED REGION ;)
    if torch.any(unknown_mask):
        # we need tensors indicating the indexes to fill the kernel
        unknown_indexes_tensor = unknown_mask\
            .unsqueeze(0)\
            .repeat(n_elements, 1)
        # mask the whole known rows to False
        unknown_indexes_tensor[known_mask] = False

        # distances between points:
        euclidean_distances = torch.cdist(
            encoded_inputs, encoded_inputs)
        # symmetrize:
        euclidean_distances = (
            euclidean_distances + euclidean_distances.t()) / 2
        # homogeneous region mask:
        homogeneous_indexes_tensor = torch.logical_or(
            known_indexes_tensor,
            unknown_indexes_tensor)
        # mask the homogeneous distances:
        euclidean_distances = torch.where(
            condition=homogeneous_indexes_tensor,
            input=torch.full(size=kernel.size,fill_value=float('inf')),
            other=euclidean_distances)

        # compute adagae kernel:
        adagae_kernel = adagae_regularization(
            euclidean_distances, sparsity)
        # heterogeneous region mask:
        heterogeneous_indexes_tensor = ~homogeneous_indexes_tensor
        # fill the heterogeneous region with the adagae kernel:
        kernel = torch.where(
            condition=heterogeneous_indexes_tensor,
            input=adagae_kernel,
            other=kernel)

       
        assert torch.all(kernel[known_mask][:, known_mask]
                         == semantic_kernel[known_mask][:, known_mask])
        assert torch.all(kernel[unknown_mask][:, known_mask]
                         == adagae_kernel[unknown_mask][:, known_mask])
        assert torch.all(kernel[known_mask][:, unknown_mask]
                         == adagae_kernel[known_mask][:, unknown_mask])
        assert torch.all(kernel[unknown_mask][:, unknown_mask]
                         == torch.zeros_like(kernel[unknown_mask][:, unknown_mask]))
        
    return kernel


def get_collaborative_kernel(
    n_elements, one_hot_masked_labels, device):

    # kernel init
    kernel = torch.ones(
        [n_elements, n_elements],
        device=device)

    # unknown mask
    unknown_mask = one_hot_masked_labels.sum(1) == 0
    # known mask:
    known_mask = ~unknown_mask
    # we need tensors indicating the indexes to fill the kernel
    known_indexes_tensor = known_mask\
        .unsqueeze(0)\
        .repeat(n_elements, 1)
    # mask the unknown rows to False
    known_indexes_tensor[unknown_mask] = False

    # build semantic kernel:
    semantic_kernel = one_hot_masked_labels @ one_hot_masked_labels.T
    # fill the known region of the kernel with the semantic kernel:
    kernel = torch.where(
        condition=known_indexes_tensor,
        input=semantic_kernel,
        other=kernel)
    return kernel


def get_centroids(
        X,
        onehot_labels,
        device):

    cluster_agg = onehot_labels.T @ X
    samples_per_cluster = onehot_labels.sum(0)
    centroids = torch.zeros_like(cluster_agg, device=device)
    missing_clusters = samples_per_cluster == 0
    existent_centroids = cluster_agg[~missing_clusters] / \
        samples_per_cluster[~missing_clusters].unsqueeze(-1)
    centroids[~missing_clusters] = existent_centroids
    assert torch.all(centroids[~missing_clusters] == existent_centroids)
    return centroids, missing_clusters

# Processor GAT Networks' Layer


class GraphAttentionV2Layer(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int,
                 is_concat: bool = False,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = True,
                 colaborative: bool = False):

        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        self.colaborative = colaborative

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_l = nn.Linear(
            in_features,
            self.n_hidden * n_heads,
            bias=False)

        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(
                in_features,
                self.n_hidden * n_heads,
                bias=False)

        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(
            self.n_hidden,
            1,
            bias=False)

        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(
            negative_slope=leaky_relu_negative_slope)

        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)

        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                h: torch.Tensor,
                adj_mat: torch.Tensor,
                masked_oh_labels: torch.Tensor):

        # Number of nodes
        n_nodes = h.shape[0]
        # unknown mask
        unknown_mask = masked_oh_labels.sum(1) == 0
        # known mask:
        known_mask = ~unknown_mask

        # The initial GAT transformations,
        # We do two linear transformations and then split it up for each head.
        g_l = self.linear_l(h).view(
            n_nodes,
            self.n_heads,
            self.n_hidden)

        g_r = self.linear_r(h).view(
            n_nodes,
            self.n_heads,
            self.n_hidden)

        # #### Calculate attention score
        g_l_repeat = g_l.repeat(
            n_nodes,
            1,
            1)

        g_r_repeat_interleave = g_r.repeat_interleave(
            n_nodes,
            dim=0)

        g_sum = g_l_repeat + g_r_repeat_interleave

        g_sum = g_sum.view(
            n_nodes,
            n_nodes,
            self.n_heads,
            self.n_hidden)

        # get energies
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        # Introduce the adj_mat
        assert adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == n_nodes
        adj_mat = adj_mat.unsqueeze(-1)
        adj_mat = adj_mat.repeat(1, 1, self.n_heads)

        if self.colaborative:
            e = e.masked_fill(adj_mat == 0, float('-inf'))
        else:
            # mask homogeneous regions
            e[known_mask][:, unknown_mask] = torch.zeros_like(
                e[known_mask][:, unknown_mask])
            e[unknown_mask][:, known_mask] = torch.zeros_like(
                e[unknown_mask][:, known_mask])

            # scale energies between 0 and 0.5:
            e = e + e.min().abs()
            e = e / (2 * e.max())

            # fill homogeneous regions
            e[known_mask][:, unknown_mask] = \
                adj_mat[known_mask][:, unknown_mask]
            e[unknown_mask][:, known_mask] = \
                adj_mat[unknown_mask][:, known_mask]

        # Normalization
        a = self.softmax(e)
        a = self.dropout(a)

        # Calculate final output for each head
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden), a.mean(dim=2)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1), a.mean(dim=2)


class FullyConnectedGATLayer(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int,
                 is_concat: bool = False,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = True):

        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_l = nn.Linear(
            in_features,
            self.n_hidden * n_heads,
            bias=False)

        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(
                in_features,
                self.n_hidden * n_heads,
                bias=False)

        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(
            self.n_hidden,
            1,
            bias=False)

        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(
            negative_slope=leaky_relu_negative_slope)

        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)

        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                h: torch.Tensor):

        # Number of nodes
        n_nodes = h.shape[0]

        # The initial GAT transformations,
        # We do two linear transformations and then split it up for each head.
        g_l = self.linear_l(h).view(
            n_nodes,
            self.n_heads,
            self.n_hidden)

        g_r = self.linear_r(h).view(
            n_nodes,
            self.n_heads,
            self.n_hidden)

        # #### Calculate attention score
        g_l_repeat = g_l.repeat(
            n_nodes,
            1,
            1)

        g_r_repeat_interleave = g_r.repeat_interleave(
            n_nodes,
            dim=0)

        g_sum = g_l_repeat + g_r_repeat_interleave

        g_sum = g_sum.view(
            n_nodes,
            n_nodes,
            self.n_heads,
            self.n_hidden)

        # get energies
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        # Normalization
        a = self.softmax(e)
        a = self.dropout(a)

        # Calculate final output for each head
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        # Concatenate the heads
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            return attn_res.mean(dim=1)


class GAT_V2_Processor(nn.Module):
    def __init__(
            self,
            h_dim,
            processor_attention_heads,
            dropout,
            device,
            colaborative=False):

        super().__init__()

        self.processing_layer = GraphAttentionV2Layer(
            in_features=h_dim,
            out_features=h_dim,
            n_heads=processor_attention_heads,
            is_concat=True,
            dropout=dropout,
            leaky_relu_negative_slope=0.2,
            share_weights=False,
            colaborative=colaborative,
        )
        self.activation = nn.LeakyReLU(
            negative_slope=0.2)

        self.device = device

    def forward(
            self,
            x,
            ohm_labels,
            sparsity):

        adj = get_kernel(
            x,
            ohm_labels,
            sparsity=sparsity,
            device=self.device)

        h, _ = self.processing_layer(
                x, adj, ohm_labels)

        h = self.activation(h)

        # Actually, from this part on, we are decoding:

        centroids, _ = get_centroids(
            h,
            ohm_labels,
            self.device)

        scores = -torch.cdist(h, centroids)

        return scores, h, adj


class GAT_V4_Processor(nn.Module):
    def __init__(
            self,
            h_dim,
            processor_attention_heads,
            dropout,
            device):

        super().__init__()

        self.processing_layer = GraphAttentionV2Layer(
            in_features=h_dim,
            out_features=h_dim,
            n_heads=processor_attention_heads,
            is_concat=True,
            dropout=dropout,
            leaky_relu_negative_slope=0.2,
            share_weights=False,
            colaborative=True,
        )

        self.relation_module = SimmilarityNet(h_dim)

        self.activation = nn.LeakyReLU(
            negative_slope=0.2)

        self.device = device

    def forward(
            self,
            x,
            ohm_labels):

        n_elements = x.size(0)
        # kernel init
        kernel = torch.ones(
            [n_elements, n_elements],
            device=self.device)

        # unknown mask
        unknown_mask = ohm_labels.sum(1) == 0
        # known mask:
        known_mask = ~unknown_mask
        # we need tensors indicating the indexes to fill the kernel
        known_indexes_tensor = known_mask\
            .unsqueeze(0)\
            .repeat(n_elements, 1)
        # mask the unknown rows to False
        known_indexes_tensor[unknown_mask] = False

        # build semantic kernel:
        semantic_kernel = ohm_labels @ ohm_labels.T
        # fill the known region of the kernel with the semantic kernel:
        kernel = torch.where(
            condition=known_indexes_tensor,
            input=semantic_kernel,
            other=kernel)

        h, kernel = self.processing_layer(
                x, kernel, ohm_labels)
        h = self.activation(h)

        # Actually, from this part on, we are decoding:

        centroids, _ = get_centroids(
            h,
            ohm_labels,
            self.device)

        n_queries = h.shape[0]
        n_centroids = centroids.shape[0]

        centroids_repeated = centroids.repeat(
            n_queries,
            1)

        queries_repeated = h.repeat_interleave(
            n_centroids,
            dim=0)

        scores = self.relation_module(
            queries_repeated, centroids_repeated)

        scores = scores.reshape(
            n_queries, n_centroids)

        return scores, h, kernel



class GAT_V5_Processor(nn.Module):
    def __init__(
            self,
            h_dim,
            processor_attention_heads,
            dropout,
            device):

        super().__init__()

        self.processing_layer = GraphAttentionV2Layer(
            in_features=h_dim,
            out_features=h_dim,
            n_heads=processor_attention_heads,
            is_concat=True,
            dropout=dropout,
            leaky_relu_negative_slope=0.2,
            share_weights=False,
            colaborative=True,
        )

        self.relation_module = SimmilarityNet(h_dim)

        self.activation = nn.LeakyReLU(
            negative_slope=0.2)

        self.device = device

    def forward(
            self,
            x,
            ohm_labels):

        n_elements = x.size(0)
        
        kernel = get_collaborative_kernel(
            n_elements, 
            ohm_labels, 
            self.device)

        h, kernel = self.processing_layer(
                x, kernel, ohm_labels)
        h = self.activation(h)

        # Actually, from this part on, we are decoding:
        centroids, missing_clusters = get_centroids(
            h,
            ohm_labels,
            self.device)

        n_queries = h.shape[0]
        
        # we take only centroids available
        # luckily compatible with our label transformation strategy
        # see (helper functions in the training notebooks).
        centroids = centroids[~missing_clusters]

        scores = 1 / (torch.cdist(h, centroids) + 1e-10)

        return scores, h, kernel


class Fully_Connected_GAT_Processor(nn.Module):
    def __init__(
            self,
            h_dim,
            max_labels_dim,
            processor_attention_heads,
            dropout,
            device):

        super().__init__()

        self.max_labels_dim = max_labels_dim

        self.processing_layer = FullyConnectedGATLayer(
            in_features=h_dim+max_labels_dim,
            out_features=h_dim,
            n_heads=processor_attention_heads,
            is_concat=True,
            dropout=dropout,
            leaky_relu_negative_slope=0.2,
            share_weights=True)

        self.activation = nn.LeakyReLU(
            negative_slope=0.2)

        self.device = device

    def forward(
            self,
            x,
            ohm_labels,
            sparsity):

        # Bruna's GNN-base-FSL strategy:
        unlabeled_mask = ohm_labels.sum(1) == 0
        unlabeled_rows = torch.ones_like(
            ohm_labels[unlabeled_mask])
        unlabeled_rows = unlabeled_rows / self.max_labels_dim
        ohm_labels[unlabeled_mask] = unlabeled_rows

        processor_inputs = torch.cat([
            x, ohm_labels], dim=1)

        h = self.activation(
            self.processing_layer(processor_inputs))

        # Actually, from this part on, we are decoding:

        centroids, _ = get_centroids(
            h,
            ohm_labels,
            self.device)

        scores = -torch.cdist(h, centroids)

        return scores, h, _


class SimmilarityNet(nn.Module):
    def __init__(
            self,
            h_dim):
        super(SimmilarityNet, self).__init__()

        self.act = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(h_dim, h_dim // 2)
        self.fc2 = nn.Linear(h_dim // 2, 1)

    def forward(self, x1, x2):
        input_to_symm = torch.abs(x1 - x2)
        symm = self.fc1(input_to_symm)
        symm = self.act(symm)
        symm = self.fc2(symm)
        return symm


class GAT_V3_Processor(nn.Module):
    def __init__(
            self,
            h_dim,
            processor_attention_heads,
            dropout,
            device,
            colaborative=False):

        super().__init__()

        self.processing_layer = GraphAttentionV2Layer(
            in_features=h_dim,
            out_features=h_dim,
            n_heads=processor_attention_heads,
            is_concat=True,
            dropout=dropout,
            leaky_relu_negative_slope=0.2,
            share_weights=False,
            colaborative=colaborative,
        )
        self.relation_module = SimmilarityNet(h_dim)
        self.activation = nn.LeakyReLU(
            negative_slope=0.2)

        self.device = device

    def forward(
            self,
            x,
            ohm_labels,
            sparsity):

        adj = get_kernel(
            x,
            ohm_labels,
            sparsity=sparsity,
            device=self.device)

        h, _ = self.processing_layer(
                x, adj, ohm_labels)

        h = self.activation(h)

        # Actually, from this part on, we are decoding:

        centroids, _ = get_centroids(
            h,
            ohm_labels,
            self.device)

        n_queries = h.shape[0]
        n_centroids = centroids.shape[0]

        centroids_repeated = centroids.repeat(
            n_queries,
            1)

        queries_repeated = h.repeat_interleave(
            n_centroids,
            dim=0)

        scores = self.relation_module(
            queries_repeated, centroids_repeated)

        scores = scores.reshape(
            n_queries, n_centroids)

        return scores, h, adj


class Relation_Network(nn.Module):
    def __init__(
            self,
            h_dim,
            device):

        super().__init__()
        """ 
        self.relation_1 = nn.Linear(h_dim*2, h_dim*2)
        self.relation_2 = nn.Linear(h_dim*2, 1)
        """
        self.relation_module = SimmilarityNet(h_dim)
        self.device = device

    def forward(
            self,
            x,
            ohm_labels):

        centroids, _ = get_centroids(
            x,
            ohm_labels,
            self.device)

        n_queries = x.shape[0]
        n_centroids = centroids.shape[0]

        centroids_repeated = centroids.repeat(
            n_queries,
            1)

        queries_repeated = x.repeat_interleave(
            n_centroids,
            dim=0)

        """
        # A somehow effective relation module:
        score_input = torch.cat(
            [centroids_repeated, queries_repeated],
            dim=1)

        score_input_swap = torch.cat(
            [queries_repeated, centroids_repeated],
            dim=1)

        scores = torch.tanh(self.relation_1(score_input))
        scores = scores + score_input_swap  # Maybe substraction would have been better here...
        scores = torch.tanh(self.relation_2(scores))
        """
        scores = self.relation_module(
            centroids_repeated, queries_repeated)

        scores = scores.reshape(
            n_queries, n_centroids)

        return scores, x


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention / self.temperature
        raw_attention = attention
        log_attention = nn.functional.log_softmax(attention, 2)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        output = torch.bmm(attention, v)
        return output, attention, log_attention, raw_attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, flag_norm=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=math.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=math.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=math.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_k))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.flag_norm = flag_norm

    def forward(self, q, k, v):
        """
        Go through the multi-head attention module.
        """
        sz_q, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_q, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_v)

        q = (
            q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)
        )  # (n*b) x lq x dk
        k = (
            k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k)
        )  # (n*b) x lk x dk
        v = (
            v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v)
        )  # (n*b) x lv x dv

        output, _, _, _ = self.attention(q, k, v)

        output = output.view(self.n_head, sz_q, len_q, self.d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_q, len_q, -1)
        )  # b x lq x (n*dv)
        resout = self.fc(output)
        output = self.dropout(resout)
        if self.flag_norm:
            output = self.layer_norm(output + residual)

        return output, resout


class FEAT_Network(nn.Module):
    def __init__(
            self,
            h_dim,
            processor_attention_heads,
            dropout,
            device):

        super().__init__()
        self.attention_module = MultiHeadAttention(
            processor_attention_heads,
            h_dim,
            h_dim,
            h_dim,
            dropout=dropout,
        )
        self.device = device

    def forward(
            self,
            x,
            ohm_labels):

        centroids, _ = get_centroids(
            x,
            ohm_labels,
            self.device)

        centroids = self.attention_module(
            centroids.unsqueeze(0),
            centroids.unsqueeze(0),
            centroids.unsqueeze(0),
        )[0][0]

        scores = -torch.cdist(x, centroids)

        return scores, x


class Super_FEAT_Network(nn.Module):
    def __init__(
            self,
            h_dim,
            processor_attention_heads,
            dropout):

        super().__init__()
        self.attention_module = MultiHeadAttention(
            processor_attention_heads,
            h_dim,
            h_dim,
            h_dim,
            dropout=dropout,
        )

    def get_centroids(
            self,
            X,
            onehot_labels):

        samples_per_cluster = onehot_labels.sum(0)
        centroids = []
        for i in range(onehot_labels.shape[1]):
            refined = self.attention_module(
                X[onehot_labels[:, i].long()].unsqueeze(0),
                X[onehot_labels[:, i].long()].unsqueeze(0),
                X[onehot_labels[:, i].long()].unsqueeze(0))[0][0]

            cluster_agg = refined.sum(0)
            if samples_per_cluster[i] > 0:
                centroid = cluster_agg / samples_per_cluster[i]
            else:
                centroid = torch.zeros_like(X[1], device=device)
            centroids.append(centroid.unsqueeze(0))

        centroids = torch.cat(centroids, 0)
        return centroids

    def forward(
            self,
            x,
            ohm_labels):

        centroids = self.get_centroids(
            x,
            ohm_labels)

        scores = -torch.cdist(x, centroids)

        return scores, x


class Adagae_Processor(nn.Module):
    def __init__(
            self,
            h_dim,
            sparsity,
            adagae_lambda,
            dropout,
            device):

        super().__init__()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sparsity = sparsity
        self.adagae_lambda = adagae_lambda
        self.device = device
        self.W = nn.Linear(
            h_dim,
            2)

    def forward(
            self,
            x,
            ohm_labels,
            spars):
        assert not torch.isnan(x.sum())

        adj = get_kernel(
            False,
            x,
            ohm_labels,
            sparsity=spars,
            adagae_lambda=self.adagae_lambda,
            semantic=True,
            normalize=False,
            device=device)

        h = self.W(x)

        g = torch.tanh(adj @ h)

        centroids, _ = get_centroids(
            g,
            ohm_labels,
            self.device)

        scores = -torch.cdist(
            g,
            centroids)

        assert not torch.isnan(adj.sum())
        return scores, g, adj


class Prototypical_Network(nn.Module):
    def __init__(
            self,
            device):

        super().__init__()
        self.device = device

    def forward(
            self,
            x,
            ohm_labels):

        centroids, _ = get_centroids(
            x,
            ohm_labels,
            self.device)

        scores = -torch.cdist(x, centroids)

        return scores, x


"""
DECODERS:
"""


class Simple_Decoder(nn.Module):
    def __init__(
            self):
        super(Simple_Decoder, self).__init__()
        """
        Our  B-type decoders output the probability
        of an input sample being known or unknown.
        """
        self.unknown_predictor = nn.Linear(1, 1)

    def forward(
            self,
            scores,
            known_class_idxs):

        # get the max score relative to an existent class
        max_known_scores = scores[:, known_class_idxs].max(1)[0].view(-1, 1)
        # predict the novelty of input samples.
        unknown_indicator = self.unknown_predictor(max_known_scores)
        unknown_indicator = torch.sigmoid(unknown_indicator)

        return unknown_indicator


class Recursive_Prototypical_Decoder(nn.Module):
    def __init__(self, device):
        super(Recursive_Prototypical_Decoder, self).__init__()
        self.device = device
        self.simple_dec = Simple_Decoder()
        self.prototypical_module = Prototypical_Network(self.device)

    def forward(self, scores, known_class_idxs, ohm_labels, unknown_mask):

        # add an indirection level to cluster-wise distances
        meta_scores, _ = self.prototypical_module(
            scores, ohm_labels)
        final_predictions = self.simple_dec(
            meta_scores[unknown_mask], known_class_idxs)
        return final_predictions


class DotProd_Decoder(nn.Module):
    def __init__(
            self,
            max_labels_dim,
            dropout,
            device):

        super(DotProd_Decoder, self).__init__()

        self.norm = nn.LayerNorm(max_labels_dim)
        self.unknown_predictor_1 = nn.Linear(
            1, 1)
        self.device = device

    def forward(
            self,
            scores,
            known_class_idxs):

        smart_key_vectors = torch.zeros(
            [1, scores.shape[1]], device=self.device)
        smart_key_vectors[0][known_class_idxs] = \
            torch.ones_like(smart_key_vectors[0][known_class_idxs])

        scores = self.norm(scores)
        unknown_indicators = scores @ smart_key_vectors.T
        unknown_indicators = self.unknown_predictor_1(unknown_indicators)

        return unknown_indicators


class Confidence_Decoder(nn.Module):
    def __init__(
            self,
            in_dim,
            dropout,
            device):

        super(Confidence_Decoder, self).__init__()

        self.score_transform = nn.Linear(
            in_features=in_dim,
            out_features=1)
        self.device = device

    def forward(
            self,
            scores):

        scores = self.score_transform(scores)
        unknown_indicators = torch.sigmoid(scores)
        return unknown_indicators


class Super_Hyper_Decoder(nn.Module):

    def __init__(
            self,
            max_labels_dim,
            dropout,
            device):

        super(Super_Hyper_Decoder, self).__init__()

        self.unknown_predictor_1 = nn.Linear(
            max_labels_dim * 2, max_labels_dim * 2)
        self.drop_1 = nn.Dropout(dropout)
        self.unknown_predictor_2 = nn.Linear(
            max_labels_dim * 2, max_labels_dim)
        self.drop_2 = nn.Dropout(dropout)
        self.unknown_predictor_3 = nn.Linear(
            max_labels_dim, 1)
        self.act = nn.LeakyReLU(0.2)
        self.device = device

    def forward(
            self,
            scores,
            known_class_idxs):

        smart_key_vectors = torch.zeros(
            [1, scores.shape[1]],
            device=self.device)

        smart_key_vectors[0][known_class_idxs] = \
            torch.ones_like(smart_key_vectors[0][known_class_idxs])

        smart_key_vectors = smart_key_vectors.repeat(
            scores.shape[0], 1)

        scores = torch.cat(
            [scores, smart_key_vectors],
            dim=1)

        unknown_indicator = self.unknown_predictor_1(scores)
        unknown_indicator = self.act(unknown_indicator)
        unknown_indicator = self.drop_1(unknown_indicator)

        unknown_indicator = self.unknown_predictor_2(unknown_indicator)
        unknown_indicator = self.act(unknown_indicator)
        unknown_indicator = self.drop_2(unknown_indicator)

        unknown_indicator = self.unknown_predictor_3(unknown_indicator)
        unknown_indicator = self.act(unknown_indicator)
        unknown_indicator = torch.sigmoid(unknown_indicator)

        return unknown_indicator


class Smart_Decoder(nn.Module):

    def __init__(
            self,
            max_labels_dim,
            dropout,
            device):

        super(Smart_Decoder, self).__init__()
        self.norm = nn.BatchNorm1d(max_labels_dim)
        self.unknown_predictor_1 = nn.Linear(
            max_labels_dim, max_labels_dim//2)
        self.drop_1 = nn.Dropout(dropout)
        self.unknown_predictor_2 = nn.Linear(
            max_labels_dim//2, 1)
        self.act = nn.LeakyReLU(0.2)
        self.device = device

    def forward(
            self,
            scores,
            known_class_idxs):

        scores[:, ~known_class_idxs] = 0
        scores = self.norm(scores)
        unknown_indicator = self.unknown_predictor_1(scores)
        unknown_indicator = self.act(unknown_indicator)
        unknown_indicator = self.drop_1(unknown_indicator)

        unknown_indicator = self.unknown_predictor_2(unknown_indicator)
        unknown_indicator = torch.sigmoid(unknown_indicator)

        return unknown_indicator


"""
ENCODERS
"""


class Encoder(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            norm=None,
            dropout=0.5
    ):
        super(Encoder, self).__init__()

        self.dropout = dropout

        if norm == "layer":
            self.norm = nn.LayerNorm(in_features)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(in_features)
        elif norm is None:
            self.norm = nn.Identity()

        # reduce to output space
        self.encoder_layer_1 = nn.Linear(
            in_features,
            out_features//2)

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)

        self.encoder_layer_2 = nn.Linear(
            out_features//2,
            out_features)

        self.act = nn.ReLU()

    def forward(
            self,
            natural_inputs):

        latent_inputs = self.norm(natural_inputs)
        latent_inputs = self.act(self.encoder_layer_1(latent_inputs))
        if self.dropout:
            latent_inputs = self.dropout_layer(latent_inputs)
        latent_inputs = self.act(self.encoder_layer_2(latent_inputs))

        return latent_inputs


"""
BOVENZI
"""
class Simple_MLP_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(Simple_MLP_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.softmax(x,1)
        return x

class FixedAnomalyDetectionModule(nn.Module):
    def __init__(self, tau_u):
        super(FixedAnomalyDetectionModule, self).__init__()
        self.tau_u = tau_u

    def forward(self, x):
        return torch.relu(x - self.tau_u)

class LearnableAnomalyDetectionModule(nn.Module):
    def __init__(self, init_treshold):
        super(LearnableAnomalyDetectionModule, self).__init__()
        # Define the threshold as a learnable parameter
        self.threshold = nn.Parameter(torch.tensor(init_treshold), requires_grad=True)
    def forward(self, x):
        return torch.relu(x - self.threshold)