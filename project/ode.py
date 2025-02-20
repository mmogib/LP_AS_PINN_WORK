import torch


def createPhi(D, A, b):
    r, n = A.shape

    def phi1(y):

        x = y[:n]
        u = y[n:]

        m = torch.clamp(u + A @ x - b, min=0.0)

        top = -(D.unsqueeze(1) + A.t() @ m)

        bottom = m - u
        # print(top.shape)
        # print(bottom.shape)
        return torch.cat((top, bottom), 0).squeeze(1)

    def phi2(y):

        x = y[:n]
        u = y[n:]

        m = torch.clamp(u + A @ x - b, min=0.0)

        top = -(D + A.t() @ m)

        bottom = m - u
        return torch.cat((top, bottom), 0)

    phi = phi1 if r == 1 else phi2

    return phi
