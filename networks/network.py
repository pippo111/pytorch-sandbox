from networks.archs.unet import Unet

def get(name):
    networks = dict(
        Unet=Unet
    )
    
    return networks[name]
