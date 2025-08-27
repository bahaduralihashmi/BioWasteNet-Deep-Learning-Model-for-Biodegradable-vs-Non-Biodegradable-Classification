def test_inference():
    from ultralytics import YOLO
    import cv2, torch

    model = YOLO(r"C:\Users\bahad\OneDrive\Desktop\ultralytics\runs\detect\Person_detect\weights\best.onnx")
    img_path = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\images\Image1.jpg"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (96, 96)) / 255.0
    img_resized = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    results = model(img_resized)
    print(results)
    assert results is not None
