from core.Nexus import Nexus, np

def im2col(x: Nexus, kh: int, kw:int, stride: int = 1, padding: int = 0)->Nexus:
    if padding > 0:
        x = x.pad2d(padding)

    N,C,H,W = x.dimension
    H_out = (H-kh)//stride + 1
    W_out = (W-kw)//stride + 1

    patches = []
    for i in range(H_out):
        for j in range(W_out):
            h_start, w_start = i*stride, j*stride
            patch = x[:, :, h_start:h_start+kh, w_start:w_start+kw]
            patch_flat = patch.reshape((N, C*kh*kw))
            patches.append(patch_flat)

    stacked = Nexus.concat([p.reshape((1,N,C*kh*kw)) for p in patches], axis=0)
    x_col = stacked.transpose(2,1,0).reshape((C*kh*kw, N*H_out*W_out))
    return x_col


def maxpool2d(x:Nexus, kernel_size: int | tuple, stride: int = None) -> Nexus:
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    if stride is None:
        stride = kh

    N,C,H,W = x.dimension
    H_out = (H-kh)//stride + 1
    W_out = (W-kw)//stride + 1

    patches = []
    for i in range(H_out):
        for j in range(W_out):
            h_s, w_s = i*stride, j*stride
            patch = x[:, :, h_s:h_s+kh, w_s:w_s+kw]

            patch_flat = patch.reshape((N,C,kh*kw))
            patches.append(patch_flat)

    stacked = Nexus.concat([p.reshape((1,N,C,kh*kw)) for p in patches], axis=0)
    stacked = stacked.transpose(1,2,0,3)

    val_max = np.max(stacked.value, axis=-1)
    arg_max = np.argmax(stacked.value, axis=-1)

    out_val = val_max.reshape((N,C,H_out, W_out))
    out = Nexus(out_val)
    out._children = {x}

    if Nexus._track_graph:
        def _backward():
            grad_stacked = np.zeros_like(stacked.value)

            n_idx, c_idx, p_idx = np.ogrid[:N, :C, :(H_out*W_out)]
            grad_reshaped = out.grads.reshape((N,C,H_out*W_out))

            grad_stacked[n_idx, c_idx, p_idx, arg_max] = grad_reshaped

            stacked.grads+=grad_stacked

        out._backward = _backward

    return out


    