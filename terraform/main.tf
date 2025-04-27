# Terraform script to launch a GPU EC2 instance for RL training (on-demand instance)

variable "region" {
  description = "AWS region to deploy the instance"
  type        = string
  default     = "us-west-1"
}

provider "aws" {
  region = var.region
}

variable "ssh_key_name" {
  description = "Name of the SSH key pair"
  type        = string
}
variable "public_key_path" {
  description = "Path to the public SSH key"
  type        = string
}
variable "subnet_id" {
  description = "Subnet ID for the instance"
  type        = string
}
variable "vpc_id" {
  description = "VPC ID for the instance"
  type        = string
}
variable "owner_tag" {
  description = "Owner tag for identifying resources"
  type        = string
}
variable "public_ip" {
  description = "Whether to associate a public IP address"
  type        = bool
  default     = false
}
variable "ami_id" {
  description = "AMI ID for the instance"
  type        = string
}
variable "instance_type" {
  description = "Instance type for the instance"
  type        = string
  default     = "g4dn.xlarge"
}

resource "aws_key_pair" "rl_key" {
  key_name   = var.ssh_key_name
  public_key = file(var.public_key_path)
  tags = {
    "owner" = var.owner_tag
  }
}

resource "aws_security_group" "rl_sg" {
  name        = "rl-sg"
  vpc_id      = var.vpc_id
  description = "Allow SSH from anywhere"

  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    cidr_blocks     = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    "owner" = var.owner_tag
  }
}

resource "aws_instance" "rl_instance" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = aws_key_pair.rl_key.key_name
  subnet_id     = var.subnet_id
  vpc_security_group_ids = [aws_security_group.rl_sg.id]
  # associate_public_ip_address = true

  instance_market_options {
    market_type = "spot"
    spot_options {
      spot_instance_type = "persistent"
      instance_interruption_behavior = "stop"
    }
  }

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
    delete_on_termination = true
    
    tags = {
        "owner" = var.owner_tag
    }
  }

  tags = {
    "Name"   = "rl-training-instance"
    "owner" = var.owner_tag
  }
}

output "instance_id" {
  value = aws_instance.rl_instance.id
}

output "private_ip" {
  value       = aws_instance.rl_instance.private_ip
  description = "Private IP of the RL training instance"
}

output "public_ip" {
  value       = aws_instance.rl_instance.public_ip
  description = "Public IP of the RL training instance"
}

output "max_spot_price" {
  value       = aws_instance.rl_instance.instance_market_options[0].spot_options[0].max_price
  description = "Spot price of the RL training instance"
}

# output "market_options" {
#   value = aws_instance.rl_instance.instance_market_options
#   description = "Market options for the RL training instance"
# }