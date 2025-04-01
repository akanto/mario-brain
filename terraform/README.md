# Terraform Setup for RL Training EC2 Instance (GPU)

This Terraform configuration launches an **on-demand GPU EC2 instance** (e.g., `g4dn.xlarge`) in your private subnet, using the AWS Deep Learning AMI (PyTorch 2.6, Ubuntu 22.04).

## Prerequisites

- Terraform installed (`brew install terraform` or from [terraform.io](https://terraform.io))
- AWS CLI configured with credentials (`aws configure`)
- VPC/Subnet created in your AWS account

## Fill in `dev.tfvars`

Create a `dev.tfvars` file in the same directory as `main.tf` with the following content, you can find dev.tfvars.example as a reference:

```hcl
ssh_key_name    = "akanto-rl-ssh"
public_key_path = "~/.ssh/id-rsa.pub"
subnet_id       = "subnet-xxxxxxxxxxxxxxxx"
vpc_id          = "vpc-xxxxxxxxxxxxxxxx"
owner_tag       = "akanto"
```

## Launch the EC2 Instance

```bash
terraform init
terraform apply -var-file=dev.tfvars
```

## Connect to the Instance

```bash
ssh ubuntu@<instance-public-ip>
```

You can inspect the nvidia-smi output to verify the GPU is available:

```bash
nvidia-smi
```

## Terminate the Instance

```bash
terraform destroy -var-file=dev.tfvars
```
