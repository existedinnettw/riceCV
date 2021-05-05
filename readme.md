

# understand google object detection API

在 Model Garden garden 裡大致有分official 和 research ，official 裡的object detection API 似乎主要是從頭開始train（還不確定model 是否會根據input image 而變），是直接用python 去執行script。由於我需要fine tune...，等多變化，自己設定執行步驟，所以不是用official 裡的東西

research 裡有各種model （Model Zoo），執行到一半的checkpoint，model config。適合做各種train 方法，步驟的更換，以jupyter notebook 為主，這才是我要用的。





* [Object Detection API 2.0, error with load checkpoints: A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. #8892](https://github.com/tensorflow/models/issues/8892)

  * finetune type 設detection 而非classification

* [“tensorflow.python.framework.errors_impl.FailedPreconditionError”  while running “model_main_tf2.py” for training object detection model  in tensorflow](https://stackoverflow.com/questions/64081214/tensorflow-python-framework-errors-impl-failedpreconditionerror-while-running)

  * model目錄內不要有checkpoint 資料夾

​      

## train



```bash
python3 "/mnt/c/Users/insleker/Google Drive/workspace/riceCV/models/research/object_detection/model_main_tf2.py" --pipeline_config_path=./pipeline.config --model_dir=. --alsologtostderr
```

## eval

```bash
python3 "/mnt/c/Users/insleker/Google Drive/workspace/riceCV/models/research/object_detection/model_main_tf2.py" --pipeline_config_path=./pipeline.config --model_dir=. --checkpoint_dir=. --alsologtostderr
```



然後開tensorboar

```bash
tensorboard --logdir=.
```





---

## export

[TensorFlow Object Detection API: Best Practices to Training, Evaluation & Deployment](https://neptune.ai/blog/tensorflow-object-detection-api-best-practices-to-training-evaluation-deployment)

```bash
python3 "/mnt/c/Users/insleker/Google Drive/workspace/riceCV/models/research/object_detection/exporter_main_v2.py" --pipeline_config_path=./pipeline.config --trained_checkpoint_dir=. --output_directory="../exported_models/?" --input_type=image_tensor
```

