# emotion_hand

## 模型下载

使用前你需要先使用

```bash
curl -o "shape_predictor_68_face_landmarks.dat" "https://objects.githubusercontent.com/github-production-release-asset-2e65be/639766419/96811955-3ef1-4b5d-9f99-4d0981cf394b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230515%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230515T125741Z&X-Amz-Expires=300&X-Amz-Signature=c63238b4431b95eeed1e4a40d139927cf3d13fb11a6220c7fac53c891f501e67&X-Amz-SignedHeaders=host&actor_id=130539515&key_id=0&repo_id=639766419&response-content-disposition=attachment%3B%20filename%3Dshape_predictor_68_face_landmarks.dat&response-content-type=application%2Foctet-stream"
```
来下载情绪识别所需要的模型

## 开启服务

打开一个终端，执行 `python emotion_hand.py` 用来启动情绪识别与手势识别服务。