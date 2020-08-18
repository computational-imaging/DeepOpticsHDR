for i in {1..4}
do
	CUDA_VISIBLE_DEVICES=1 python3 -u demo_function.py \
	 --mynet_params "PretrainedNetworks/RealPSF_Network/logs/model_step_515000.npz"\
	 --suffix "RealPSF_Network" \
	 --TestNumber $i
done
