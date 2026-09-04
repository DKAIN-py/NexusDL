from core.Nexus import Nexus, np

def im2col(x: Nexus, kh: int, kw:int, stride: int = 1, padding: int = 0)->Nexus:
    if padding > 0:
        x = x.pad2d(padding)
    
    N,C,H,W = x.dimension
    H_out = (H-kh)//stride + 1
    W_out = (W-kw)//stride + 1

    x_val = x.value
    shape = (N, C, H_out, W_out, kh, kw)
    strides = (
        x_val.strides[0],
        x_val.strides[1],
        x_val.strides[2]*stride,
        x_val.strides[3]*stride,
        x_val.strides[2],
        x_val.strides[3],
    )
    patches = np.lib.stride_tricks.as_strided(x_val, shape=shape, strides=strides)
    x_col = patches.transpose(1,4,5,0,2,3).reshape((C*kh*kw, N*H_out*W_out))

    out = Nexus(x_col)
    out._children = {x}

    if Nexus._track_graph:
        def _backward():
            grad_col = out.grads.reshape(C, kh, kw, N, H_out, W_out)
            grad_patches = grad_col.transpose(3,0,1,2,4,5)

            dx = np.zeros_like(x_val)
            for i in range(kh):
                for j in range(kw):
                    dx[:,:,i:i+H_out*stride:stride, j:j+W_out*stride:stride] += grad_patches[:,:,i,j,:,:]

            x.grads += x._handle_broadcast(dx, x.dimension)

        out._backward = _backward

    return out

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


    