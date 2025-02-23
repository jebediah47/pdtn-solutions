import os
import torch
import torchvision
from gdown import download
from zipfile import ZipFile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Χρησιμοποιείται:", device)


def download_and_extract(id: str, filename: str, extracted_folder: str):
    # Κατεβάζουμε το αρχείο zip εάν δεν υπάρχει ήδη
    if not os.path.exists(filename):
        print("Γίνεται φόρτωση του αρχείου", filename)
        download(id=id, output=filename, quiet=False)

    # Εξαγωγή του αρχείου zip εάν ο φάκελος δεν υπάρχει ήδη
    if not os.path.exists(extracted_folder):
        print("Γίνεται αποσυμπίεση των αρχείων...")
        with ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(extracted_folder)


main_folder = "knit"
download_and_extract("1FuQN05LKMaexSESklGxRr5av1EERGU5w", "knit.zip", main_folder)


def readImage(filename: str, S: int = 800):
    with open(filename, "rb") as f:
        image_bytes = f.read()

    image_tensor = torchvision.io.decode_image(
        torch.tensor(list(image_bytes), device="cpu").byte(),
        mode=torchvision.io.ImageReadMode.GRAY,
    )

    image_tensor = 1 - image_tensor.float() / 255.0
    resize_transform = torchvision.transforms.Resize((S, S))
    return image_tensor.squeeze().to(device)


def showImage(I):
    return torchvision.transforms.functional.to_pil_image((1 - I).clamp(0, 1))


# target = readImage('knit/star.jpg')
# target = readImage('knit/circles.jpg')
# target = readImage('knit/world.jpg')
target = readImage("knit/eye.jpg")
showImage(target)

N = 200
S = target.shape[0]
R = S // 2

i, j = torch.meshgrid(
    torch.arange(-R, R, device=device),
    torch.arange(-R, R, device=device),
    indexing="ij",
)
mask = (i**2 + j**2) <= R**2
points = torch.stack((i[mask], j[mask]), dim=-1).float()

angles = torch.arange(N, dtype=torch.float32, device=device) * torch.pi / N
rotation_matrix = torch.stack(
    [torch.cos(angles), -torch.sin(angles)], dim=0
)  # shape (2, N)
rotated_points = points @ rotation_matrix  # shape (num_points, N)

rotated_points_line = N * (
    torch.acos(torch.clamp(rotated_points / R, -1.0, 1.0)) / torch.pi
) - (torch.arange(N, device=device) % 2)
closest_line = torch.round(rotated_points_line / 2).clamp(0, N // 2 - 1).int()

rows = torch.stack(
    [
        R * torch.cos((2 * torch.arange(N // 2, device=device) + 0) * torch.pi / N),
        R * torch.cos((2 * torch.arange(N // 2, device=device) + 1) * torch.pi / N),
    ],
    dim=1,
)


def linesToImage(lines: torch.Tensor) -> torch.Tensor:
    L = lines[closest_line, torch.arange(N, device=device)].clamp(0, 1)
    diff = torch.abs(
        rotated_points - rows[closest_line, torch.arange(N, device=device) % 2]
    )
    L = L * (diff <= 1).float()
    I = torch.zeros((S, S), device=device)
    I[mask] = L.sum(dim=-1)
    return I


lines = torch.ones((N // 2, N), requires_grad=True, device=device)
image_out = linesToImage(lines)

target = target.to(device)
mask = mask.to(device)
points = points.to(device)
angles = angles.to(device)
rotated_points = rotated_points.to(device)

rotated_points_line = rotated_points_line.to(device)
closest_line = closest_line.to(device)
rows = rows.to(device)

lines = lines.to(device)
lines.requires_grad_()

optimizer = torch.optim.AdamW([lines], lr=0.1)
loss_fn = torch.nn.MSELoss()

num_iterations = 300

print("Έναρξη βελτιστοποίησης...")
for iter in range(num_iterations):
    optimizer.zero_grad()
    output_image = linesToImage(lines)
    loss = loss_fn(output_image, target)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        lines.clamp_(0, 100)
    if iter % 10 == 0:
        print(f"Iteration {iter}, Loss: {loss.item():.6f}")

print("Βελτιστοποίηση ολοκληρώθηκε.")


def getWeights(lines):
    return "\n".join(
        [
            " ".join(map(str, x))
            for x in ((100 * lines.T).clamp(0, 100.0).int()).tolist()
        ]
    )


getWeights(lines)
