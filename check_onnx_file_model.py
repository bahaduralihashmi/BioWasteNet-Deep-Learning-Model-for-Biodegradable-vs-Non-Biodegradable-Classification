def test_inference():
    from ultralytics import YOLO
    import cv2, torch

    model = YOLO(r"C:\Users\bahad\OneDrive\Desktop\ultralytics\runs\detect\Person_detect\weights\best.onnx")
    img_path = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\images\Image1.jpg"

    # Load grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize and normalize
    img_resized = cv2.resize(img, (96, 96)) / 255.0

    # Add dimensions for [1, 1, H, W]
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Run inference
    results = model(img_tensor)
    print(results)
    assert results is not None
