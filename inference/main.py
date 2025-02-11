import cv2
import torch
from torchvision import transforms
import numpy as np

from model import squeezenet , mobilenet

label_dict = {0: "female", 1: "male"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = squeezenet.Custom_squeezenet(num_class=2)
model.load_state_dict(torch.load("yahan path location", map_location=device))
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # For SqueezeNet, typically 224x224 is used
    transforms.ToTensor(),
])

def infer_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(rgb_frame)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = output.view(output.size(0), -1)
        prob = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(prob, dim=1)

    return label_dict[pred_idx.item()], confidence.item()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("webcam nhi khul raha")
        return

    print("q dba quit krne ke liye bsdk")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_label, conf = infer_frame(frame)

        text = f"{pred_label} ({conf * 100:.1f}%)"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()