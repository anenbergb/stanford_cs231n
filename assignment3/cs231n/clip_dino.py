import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import tensorflow_datasets as tfds
from torchvision import transforms as T
import cv2
from tqdm.auto import tqdm


def get_similarity_no_loop(text_features, image_features):
    """
    Computes the pairwise cosine similarity between text and image feature vectors.

    Args:
        text_features (torch.Tensor): A tensor of shape (N, D).
        image_features (torch.Tensor): A tensor of shape (M, D).

    Returns:
        torch.Tensor: A similarity matrix of shape (N, M), where each entry (i, j)
        is the cosine similarity between text_features[i] and image_features[j].
    """
    similarity = None
    ############################################################################
    # TODO: Compute the cosine similarity. Do NOT use for loops.               #
    ############################################################################
    text_norm = torch.sqrt(torch.sum(text_features**2, dim=1, keepdim=True))
    image_norm = torch.sqrt(torch.sum(image_features**2, dim=1, keepdim=True))
    text_features_norm = text_features / text_norm
    image_features_norm = image_features / image_norm
    similarity = torch.matmul(text_features_norm, image_features_norm.T)

    # similarity = nn.functional.cosine_similarity(
    #     text_features[:, None], image_features, dim=-1
    # )

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return similarity


@torch.no_grad()
def clip_zero_shot_classifier(clip_model, clip_preprocess, images,
                              class_texts, device):
    """Performs zero-shot image classification using a CLIP model.

    Args:
        clip_model (torch.nn.Module): The pre-trained CLIP model for encoding
            images and text.
        clip_preprocess (Callable): A preprocessing function to apply to each
            image before encoding.
        images (List[np.ndarray]): A list of input images as NumPy arrays
            (H x W x C) uint8.
        class_texts (List[str]): A list of class label strings for zero-shot
            classification.
        device (torch.device): The device on which computation should be
            performed. Pass text_tokens to this device before passing it to
            clip_model.

    Returns:
        List[str]: Predicted class label for each image, selected from the
            given class_texts.
    """
    
    pred_classes = []

    ############################################################################
    # TODO: Find the class labels for images.                                  #
    ############################################################################
    class_tokens = clip.tokenize(class_texts).to(device) # (N,77)
    with torch.no_grad():
        class_text_feats = clip_model.encode_text(class_tokens) # (N,D)

    processed_images = [
        clip_preprocess(Image.fromarray(img)).unsqueeze(0)
        for img in images
    ]
    images_tensor = torch.cat(processed_images, dim=0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(images_tensor) # (N,D)

    similarity = get_similarity_no_loop(class_text_feats, image_features) # (N,N)
    similarity = similarity.detach().cpu().numpy()
    image2class_match = np.argmax(similarity, axis=0) # argmax along the text dimension
    pred_classes = [class_texts[i] for i in image2class_match]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return pred_classes
  

class CLIPImageRetriever:
    """
    A simple image retrieval system using CLIP.
    """
    
    @torch.no_grad()
    def __init__(self, clip_model, clip_preprocess, images, device):
        """
        Args:
          clip_model (torch.nn.Module): The pre-trained CLIP model.
          clip_preprocess (Callable): Function to preprocess images.
          images (List[np.ndarray]): List of images as NumPy arrays (H x W x C).
          device (torch.device): The device for model execution.
        """
        ############################################################################
        # TODO: Store all necessary object variables to use in retrieve method.    #
        # Note that you should process all images at once here and avoid repeated  #
        # computation for each text query. You may end up NOT using the above      #
        # similarity function for most compute-optimal implementation.#
        ############################################################################
        self.clip_model = clip_model
        self.device = device

        self.images = images
        processed_images = [
            clip_preprocess(Image.fromarray(img)).unsqueeze(0)
            for img in images
        ]
        images_tensor = torch.cat(processed_images, dim=0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(images_tensor) # (N,D)
        
        # normalized image features
        self.image_features = image_features / torch.sqrt(torch.sum(image_features**2, dim=1, keepdim=True))


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        pass
    
    @torch.no_grad()
    def retrieve(self, query: str, k: int = 2):
        """
        Retrieves the indices of the top-k images most similar to the input text.
        You may find torch.Tensor.topk method useful.

        Args:
            query (str): The text query.
            k (int): Return top k images.

        Returns:
            List[int]: Indices of the top-k most similar images.
        """
        top_indices = []
        ############################################################################
        # TODO: Retrieve the indices of top-k images.                              #
        ############################################################################
        class_tokens = clip.tokenize(query).to(self.device) # (1,77)
        with torch.no_grad():
            class_text_feats = self.clip_model.encode_text(class_tokens) # (1,D)

        class_text_feats /= torch.sqrt(torch.sum(class_text_feats**2, dim=1, keepdim=True))
        # (1,D) x (D,N) = (1,N)
        similarity = torch.matmul(class_text_feats, self.image_features.T).flatten()
        _, top_indices = similarity.topk(k)
        top_indices = top_indices.detach().cpu().tolist()

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return top_indices

  
class DavisDataset:
    def __init__(self):
        self.davis = tfds.load('davis/480p', split='validation', as_supervised=False)
        self.img_tsfm = T.Compose([
            T.Resize((480, 480)), T.ToTensor(),
            T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        
      
    def get_sample(self, index):
        assert index < len(self.davis)
        ds_iter = iter(tfds.as_numpy(self.davis))
        for i in range(index+1):
            video = next(ds_iter)
        frames, masks = video['video']['frames'], video['video']['segmentations']
        print(f"video {video['metadata']['video_name'].decode()}  {len(frames)} frames")
        return frames, masks
    
    def process_frames(self, frames, dino_model, device):
        res = []
        for f in frames:
            f = self.img_tsfm(Image.fromarray(f))[None].to(device)
            with torch.no_grad():
              tok = dino_model.get_intermediate_layers(f, n=1)[0]
            res.append(tok[0, 1:])

        res = torch.stack(res)
        return res
    
    def process_masks(self, masks, device):
        res = []
        for m in masks:
            m = cv2.resize(m, (60,60), cv2.INTER_NEAREST)
            res.append(torch.from_numpy(m).long().flatten(-2, -1))
        res = torch.stack(res).to(device)
        return res
    
    def mask_frame_overlay(self, processed_mask, frame):
        H, W = frame.shape[:2]
        mask = processed_mask.detach().cpu().numpy()
        mask = mask.reshape((60, 60))
        mask = cv2.resize(
            mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = create_segmentation_overlay(mask, frame.copy())
        return overlay
        


def create_segmentation_overlay(segmentation_mask, image, alpha=0.5):
    """
    Generate a colored segmentation overlay on top of an RGB image.

    Parameters:
        segmentation_mask (np.ndarray): 2D array of shape (H, W), with class indices.
        image (np.ndarray): 3D array of shape (H, W, 3), RGB image.
        alpha (float): Transparency factor for overlay (0 = only image, 1 = only mask).

    Returns:
        np.ndarray: Image with segmentation overlay (shape: (H, W, 3), dtype: uint8).
    """
    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"

    # Generate deterministic colors for each class using a fixed colormap
    def generate_colormap(n):
        np.random.seed(42)  # For determinism
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # Create a color image for the segmentation mask
    seg_color = colormap[segmentation_mask]  # shape: (H, W, 3)

    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

    return overlay


def compute_iou(pred, gt, num_classes):
    """Compute the mean Intersection over Union (IoU)."""
    iou = 0
    for ci in range(num_classes):
        p = pred == ci
        g = gt == ci
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes


class DINOSegmentation:
    def __init__(self, device, num_classes: int, inp_dim : int = 384):
        """
        Initialize the DINOSegmentation model.

        This defines a simple neural network designed to  classify DINO feature
        vectors into segmentation classes. It includes model initialization,
        optimizer, and loss function setup.

        Args:
            device (torch.device): Device to run the model on (CPU or CUDA).
            num_classes (int): Number of segmentation classes.
            inp_dim (int, optional): Dimensionality of the input DINO features.
        """

        ############################################################################
        # TODO: Define a very lightweight pytorch model, optimizer, and loss       #
        # function to train classify each DINO feature vector into a seg. class.   #
        # It can be a linear layer or two layer neural network.                    #
        ############################################################################
        self.hidden_dim = 512
        self.classifier = nn.Sequential(
            nn.Linear(inp_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes)
        )
        self.device = device

        self.optimizer = torch.optim.AdamW(self.classifier.parameters(), lr = 2e-4, weight_decay=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.classifier.to(device)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def train(self, X_train, Y_train, num_iters=500):
        """Train the segmentation model using the provided training data.

        Args:
            X_train (torch.Tensor): Input feature vectors of shape (N, D).
            Y_train (torch.Tensor): Ground truth labels of shape (N,).
            num_iters (int, optional): Number of optimization steps.

        N could be the tokens for a single image.
        So an input image of size 480x480 could be broken into 60^2 patches of size 8x8
        So N could be N = 3600
        Each of these 8x8 patches receives a segmentation label, which is why there are N labels.
        
        """
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        for i in range(num_iters):
            self.optimizer.zero_grad()
            logits = self.classifier(X_train)
            loss = self.criterion(logits, Y_train)
            loss.backward()
            self.optimizer.step()
            if (i+1) % 100 == 0:
                print(f"Training [{i+1}/{num_iters}] loss: {loss.detach().cpu().item():.2f}")

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
    @torch.no_grad()
    def inference(self, X_test):
        """Perform inference on the given test DINO feature vectors.

        Args:
            X_test (torch.Tensor): Input feature vectors of shape (N, D).

        Returns:
            torch.Tensor of shape (N,): Predicted class indices.
        """
        with torch.no_grad():
            pred_logits = self.classifier(X_test) # (N,num_classes)
            pred_prob = torch.softmax(pred_logits, dim=-1)  # (N,num_classes)
            pred_classes = torch.argmax(pred_prob, dim=-1) # (N,)

        return pred_classes