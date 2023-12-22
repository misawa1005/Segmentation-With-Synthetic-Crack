#!/bin/sh
cd GroundedSAM/GroundingDINO
poetry run python setup.py build
poetry run python setup.py install
cd ../..
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth