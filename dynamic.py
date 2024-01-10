import onnx
mp = onnx.load_model('./best.onnx')
mp.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
mp.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '?'
onnx.save(mp, './best.onnx')