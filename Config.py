# 함수들을 실행시키면 config라는 ConfigDict안에 parameter 특성들이 저장 됌.
import ml_collections

def get_ViT_B() :
    config = ml_collections.ConfigDict()
    config.num_layers = 12
    config.hid_dim = 768
    config.ff_dim = 3072
    config.n_heads = 12
    return config

def get_ViT_L() :
    config = ml_collections.ConfigDict()
    config.num_layers = 24
    config.hid_dim = 1024
    config.ff_dim = 4096
    config.n_heads = 16
    return config

def get_ViT_H() :
    config = ml_collections.ConfigDict()
    config.num_layers = 32
    config.hid_dim = 1280
    config.ff_dim = 5120
    config.n_heads = 16
    return config


def get_ViT_B_16():
    config = get_ViT_B()
    config.patch_size = 16
    return config

def get_ViT_B_32():
    config = get_ViT_B()
    config.patch_size = 32
    return config

def get_ViT_L_16():
    config = get_ViT_L()
    config.patch_size = 16
    return config

def get_ViT_L_32():
    config = get_ViT_L()
    config.patch_size = 32
    return config

def get_ViT_H_16():
    config = get_ViT_H()
    config.patch_size = 16
    return config

def get_ViT_H_32():
    config = get_ViT_H()
    config.patch_size = 32
    return config

def for_test_ViT_B_14():
    config = get_ViT_B()
    config.patch_size = 14
    return config
