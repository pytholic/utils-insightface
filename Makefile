# pass the network argument
# Example:  make network=efficientnet-lite convert

convert: 
	python3 torch2onnx.py --input /home/jovyan/haseeb-data/insightface/work_dirs/mbf-GNAP-emore-512/model.pt --output model.onnx --network "$(network)" --simplify True
train:
	torchrun --nproc_per_node=2  train_v2.py configs/custom.py
