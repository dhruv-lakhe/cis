import base64
import hashlib
import io
import os
import tkinter as tk
import traceback
from tkinter import Button, Label, filedialog, messagebox

import cv2
import numpy as np
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from PIL import Image, ImageTk

# Load the pre-trained face detection model
face_net = cv2.dnn.readNetFromCaffe('face_files/deploy.prototxt', 'face_files/res10_300x300_ssd_iter_140000.caffemodel')

# Generate a key from the face image
def generate_key_from_face(face_image):
    face_image = cv2.resize(face_image, (128, 128))
    face_bytes = face_image.flatten().tobytes()
    key = hashlib.sha256(face_bytes).digest()
    print(f"Generated key: {key.hex()}")  # Debug print
    return key

# AES encryption


# Detect faces in a frame
def detect_faces(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    confidences = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
            confidences.append(confidence)
    return faces, confidences

# Function to show the live camera feed in the GUI
def show_frame():
    ret, frame = cap.read()
    if not ret:
        return

    faces, confidences = detect_faces(frame)

    for (box, confidence) in zip(faces, confidences):
        (startX, startY, endX, endY) = box
        text = f"{confidence * 100:.2f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(image=frame_pil)

    camera_label.imgtk = frame_tk
    camera_label.configure(image=frame_tk)
    camera_label.after(10, show_frame)

# Function to capture the face image
def capture_face():
    global face_key
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image")
        return

    faces, confidences = detect_faces(frame)
    if len(faces) > 0:
        (startX, startY, endX, endY) = faces[0]
        face_image = frame[startY:endY, startX:endX]
        face_key = generate_key_from_face(face_image)
        
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tk = ImageTk.PhotoImage(face_pil)
        captured_face_label.config(image=face_tk)
        captured_face_label.image = face_tk
        
        # Save the face key to a file
        with open("face_key.bin", "wb") as f:
            f.write(face_key)
        
        messagebox.showinfo("Success", "Face captured and key generated")
        select_image_button.config(state=tk.NORMAL)
    else:
        messagebox.showerror("Error", "No face detected")

# Function to load the face key
def load_face_key():
    global face_key
    if os.path.exists("face_key.bin"):
        with open("face_key.bin", "rb") as f:
            face_key = f.read()
        print(f"Loaded key: {face_key.hex()}")  # Debug print
        return True
    return False

# Function to select an image
def select_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
    if selected_image_path:
        selected_image = Image.open(selected_image_path)
        selected_image.thumbnail((250, 250))
        selected_image_tk = ImageTk.PhotoImage(selected_image)
        selected_image_label.config(image=selected_image_tk)
        selected_image_label.image = selected_image_tk
        
        encrypt_image_button.config(state=tk.NORMAL)

# Function to encrypt the selected image
# AES encryption
def aes_encrypt(key, data):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data) + encryptor.finalize()
    return iv + encrypted_data

# AES decryption
def aes_decrypt(key, encrypted_data):
    iv = encrypted_data[:16]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(encrypted_data[16:]) + decryptor.finalize()

# Function to encrypt the selected image
def encrypt_image():
    if not face_key or not selected_image_path:
        messagebox.showerror("Error", "Please capture a face and select an image first")
        return

    with open(selected_image_path, "rb") as f:
        image_data = f.read()

    encrypted_data = aes_encrypt(face_key, image_data)
    
    # Save the encrypted data
    save_path = filedialog.asksaveasfilename(defaultextension=".enc", filetypes=[("Encrypted files", "*.enc")])
    if save_path:
        with open(save_path, "wb") as f:
            f.write(encrypted_data)
        print(f"Encrypted with key: {face_key.hex()}")  # Debug print
        messagebox.showinfo("Success", f"Encrypted image saved to {save_path}")
    else:
        messagebox.showerror("Error", "Failed to save the encrypted image")

# Function to decrypt an image
def decrypt_image():
    global face_key
    if not face_key:
        if not load_face_key():
            messagebox.showerror("Error", "No face key available. Please capture a face first.")
            return
    
    encrypted_path = filedialog.askopenfilename(filetypes=[("Encrypted files", "*.enc")])
    if not encrypted_path:
        return

    try:
        # Read the encrypted data
        with open(encrypted_path, "rb") as f:
            encrypted_data = f.read()

        print(f"Decrypting with key: {face_key.hex()}")  # Debug print
        print(f"Encrypted data length: {len(encrypted_data)}")

        # Decrypt the data
        decrypted_data = aes_decrypt(face_key, encrypted_data)
        
        print(f"Decrypted data length: {len(decrypted_data)}")
        print(f"First 50 bytes of decrypted data: {decrypted_data[:50]}")

        # Try to create a PIL Image from the decrypted data
        try:
            decrypted_image = Image.open(io.BytesIO(decrypted_data))
            print("Successfully created PIL Image from decrypted data")
        except Exception as e:
            print(f"Failed to create PIL Image: {str(e)}")
            raise

        # Save the decrypted image
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if save_path:
            decrypted_image.save(save_path)
            print(f"Successfully saved image to {save_path}")

            # Display the decrypted image
            decrypted_image.thumbnail((250, 250))
            decrypted_tk = ImageTk.PhotoImage(decrypted_image)
            decrypted_image_label.config(image=decrypted_tk)
            decrypted_image_label.image = decrypted_tk
            
            messagebox.showinfo("Success", f"Decrypted image saved to {save_path}")
        else:
            messagebox.showerror("Error", "Failed to save the decrypted image")
    except Exception as e:
        error_message = f"Decryption failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        messagebox.showerror("Error", error_message)
        print(error_message)

# Set up the GUI window
root = tk.Tk()
root.title("Face-based Image Encryption and Decryption")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set up GUI elements
camera_label = Label(root)
camera_label.grid(row=0, column=0, padx=10, pady=10)

captured_face_label = Label(root, text="Captured Face")
captured_face_label.grid(row=0, column=1, padx=10, pady=10)

selected_image_label = Label(root, text="Selected Image")
selected_image_label.grid(row=0, column=2, padx=10, pady=10)

encrypted_image_label = Label(root, text="Encrypted Image")
encrypted_image_label.grid(row=0, column=3, padx=10, pady=10)

decrypted_image_label = Label(root, text="Decrypted Image")
decrypted_image_label.grid(row=0, column=4, padx=10, pady=10)

capture_button = Button(root, text="Capture Face", command=capture_face)
capture_button.grid(row=1, column=0, padx=10, pady=10)

select_image_button = Button(root, text="Select Image", command=select_image, state=tk.DISABLED)
select_image_button.grid(row=1, column=1, padx=10, pady=10)

encrypt_image_button = Button(root, text="Encrypt Image", command=encrypt_image, state=tk.DISABLED)
encrypt_image_button.grid(row=1, column=2, padx=10, pady=10)

decrypt_image_button = Button(root, text="Decrypt Image", command=decrypt_image)
decrypt_image_button.grid(row=1, column=3, padx=10, pady=10)

# Initialize global variables
if load_face_key():
    select_image_button.config(state=tk.NORMAL)
selected_image_path = None

# Start displaying the live camera feed
show_frame()

# Run the GUI loop
root.mainloop()

# Release the webcam when the window is closed
cap.release()
cv2.destroyAllWindows()