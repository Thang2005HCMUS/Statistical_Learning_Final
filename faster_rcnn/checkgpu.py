import torch

def main():
    # Check CUDA availability
    if not torch.cuda.is_available():  # Valid API [InlineCitation-1-torch.cuda.is_available — PyTorch 2.12 documentation](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.is_available.html)
        print("CUDA is NOT available. Running on CPU.")
        return

    print("CUDA is available!")

    # Number of GPUs
    try:
        num_gpus = torch.cuda.device_count()  # Valid API [InlineCitation-2-python - How do I check if PyTorch is using the GPU? - Stack Overflow](https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu) [InlineCitation-3-PyTorch: Printing Available GPUs — codegenes.net](https://www.codegenes.net/blog/pytorch-print-available-gpus/)
        print(f"Number of available GPUs: {num_gpus}")
    except Exception as e:
        print(f"Error retrieving GPU count: {e}")
        return

    # List available GPUs
    for idx in range(num_gpus):
        try:
            name = torch.cuda.get_device_name(idx)  # Valid API [InlineCitation-2-python - How do I check if PyTorch is using the GPU? - Stack Overflow](https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu) [InlineCitation-3-PyTorch: Printing Available GPUs — codegenes.net](https://www.codegenes.net/blog/pytorch-print-available-gpus/)
            print(f"GPU {idx}: {name}")
        except Exception as e:
            print(f"Error retrieving name for GPU {idx}: {e}")

    # Display memory usage for GPU 0
    try:
        allocated = torch.cuda.memory_allocated(0)  # Valid API [InlineCitation-2-python - How do I check if PyTorch is using the GPU? - Stack Overflow](https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu)
        reserved = torch.cuda.memory_reserved(0)    # Valid API [InlineCitation-2-python - How do I check if PyTorch is using the GPU? - Stack Overflow](https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu)
        print("Memory Usage on GPU 0:")
        print(f"  Allocated: {allocated / 1024**3:.2f} GB")
        print(f"  Reserved:  {reserved / 1024**3:.2f} GB")
    except Exception as e:
        print(f"Error retrieving memory info: {e}")

if __name__ == "__main__":
    main()
