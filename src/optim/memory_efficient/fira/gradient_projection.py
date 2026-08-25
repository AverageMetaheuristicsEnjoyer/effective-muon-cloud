import torch


class GradientProjector:
    def __init__(
        self, rank, verbose=False, update_proj_gap=200, alpha=1.0, proj_type="std", proj_alg="svd"
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.alpha = alpha
        self.ortho_matrix = None
        self.proj_type = proj_type

        self.proj_alg = proj_alg
        assert self.proj_alg in ["svd", "power_iteration"]
        assert not (self.proj_type == "full" and self.proj_alg == "power_iteration")

    def project(self, full_rank_grad, iter):

        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right"
                    ) if (self.proj_alg == "svd" or self.ortho_matrix is None) else self._get_orthogonal_matrix_power_it(
                        full_rank_grad, self.ortho_matrix, type="right"
                    )
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left"
                    ) if (self.proj_alg == "svd" or self.ortho_matrix is None) else self._get_orthogonal_matrix_power_it(
                        full_rank_grad, self.ortho_matrix, type="left"
                    )
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left"
                    ) if (self.proj_alg == "svd" or self.ortho_matrix is None) else self._get_orthogonal_matrix_power_it(
                        full_rank_grad, self.ortho_matrix, type="left"
                    )
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right"
                    ) if (self.proj_alg == "svd" or self.ortho_matrix is None) else self._get_orthogonal_matrix_power_it(
                        full_rank_grad, self.ortho_matrix, type="right"
                    )
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == "right":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="right"
                ) if (self.proj_alg == "svd" or self.ortho_matrix is None) else self._get_orthogonal_matrix_power_it(
                    full_rank_grad, self.ortho_matrix, type="right"
                )   
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == "left":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="left"
                ) if (self.proj_alg == "svd" or self.ortho_matrix is None) else self._get_orthogonal_matrix_power_it(
                    full_rank_grad, self.ortho_matrix, type="left"
                )
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == "full":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="full"
                )
            low_rank_grad = (
                torch.matmul(self.ortho_matrix[0].t(), full_rank_grad)
                @ self.ortho_matrix[1].t()
            )

        return low_rank_grad

    def project_back(self, low_rank_grad):

        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == "reverse_std":
            if (
                low_rank_grad.shape[0] <= low_rank_grad.shape[1]
            ):  # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == "right":
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == "left":
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == "full":
            full_rank_grad = (
                torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
            )

        return full_rank_grad * self.alpha

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
        
        if rank > 0:
            U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)
        else:
            U, s, Vh = torch.zeros(matrix.shape[0], matrix.shape[0], device=matrix.device), 0, torch.zeros(matrix.shape[1], matrix.shape[1], device=matrix.device)

        # make the smaller matrix always to be orthogonal matrix
        if type == "right":
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == "left":
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type == "full":
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError("type should be left, right or full")

    def _get_orthogonal_matrix_power_it(self, matrix, init, type, intermediate_orthogonalization=False): 
        if init is None:
            init = self.ortho_matrix

        if type == 'right':
            U = matrix @ init.t()
            if intermediate_orthogonalization : # Not necessary for computing right singular vectors
                U = Gram_Schmidt(U)
            projection_map = matrix.t() @ U
            del U

            projection_map = Gram_Schmidt(projection_map)
            return projection_map.t()

        elif type == 'left':
            V = matrix.t() @ init
            if intermediate_orthogonalization : # Not necessary for computing left singular vectors
                V = Gram_Schmidt(V)
            projection_map = matrix @ V
            del V

            projection_map = Gram_Schmidt(projection_map)
            return projection_map

        else:
            raise ValueError
        
def Gram_Schmidt(matrix):
    original_type = matrix.dtype #torch.linalg.qr doesn't support helf precision types such as torch.bfloat16
    matrix, _ = torch.linalg.qr(matrix.to(dtype=torch.float32))
    matrix = matrix.to(dtype=original_type)

    return matrix