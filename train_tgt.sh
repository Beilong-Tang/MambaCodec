## redirec the output to a date file 
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
filename="./log/${current_time}.txt"
python -u train.py --config_path /DKUdata/tangbl/MambaTransformer/config/base_tgt.yaml \
   --name base_target --ckpt_path /DKUdata/tangbl/MambaTransformer/ckpt --device cuda:4 \
   > $filename 2>&1