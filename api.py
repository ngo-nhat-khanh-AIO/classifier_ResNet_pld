import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# === Định nghĩa lại kiến trúc ResNet tùy chỉnh ===
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x += self.downsample(shortcut)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, residual_block, n_blocks_lst, n_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.create_layer(residual_block, 64, 64, n_blocks_lst[0], 1)
        self.conv3 = self.create_layer(residual_block, 64, 128, n_blocks_lst[1], 2)
        self.conv4 = self.create_layer(residual_block, 128, 256, n_blocks_lst[2], 2)
        self.conv5 = self.create_layer(residual_block, 256, 512, n_blocks_lst[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, n_classes)

    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        blocks = []
        first_block = residual_block(in_channels, out_channels, stride)
        blocks.append(first_block)

        for idx in range(1, n_blocks):
            block = residual_block(out_channels, out_channels, stride=1)
            blocks.append(block)

        block_sequential = nn.Sequential(*blocks)

        return block_sequential

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# === Load model và trọng số ===
MODEL_PATH = "model_pld.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    n_classes = 3  # Số class (Early Blight, Late Blight, Healthy)
    model = ResNet(
        residual_block=ResidualBlock,
        n_blocks_lst=[2, 2, 2, 2],
        n_classes=n_classes
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Sửa lại hàm này
    model.eval()
    return model

model = load_model()

# === Tiền xử lý ảnh ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Khởi tạo API ===
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Tiền xử lý ảnh
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]  # Xác suất các lớp
        _, predicted_class = torch.max(output, 1)

    # Mapping label
    label_mapping = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}
    predicted_label = label_mapping[int(predicted_class.item())]

    # Xác định "Có bệnh" hay "Không có bệnh"
    if predicted_class.item() in [0, 1]:
        diagnosis = "Có bệnh"
    else:
        diagnosis = "Không có bệnh"

    return {
        "predicted_class": predicted_class.item(),
        "label": predicted_label,
        "diagnosis": diagnosis,
        "probabilities": probabilities.tolist()  # Chuyển xác suất thành danh sách JSON
    }
