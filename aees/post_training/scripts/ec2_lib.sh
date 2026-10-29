# EC2 helpers shared by the laptop scripts (proto/ec2_up.sh, scripts/ec2_sandboxes.sh). Source it; don't run it.
# Uses the laptop's AWS CLI profile (AWS_PROFILE); nothing is copied to the instances.

ec2_identity() {      # prints who we are and, if readable, the budget headroom (the auto-stop spend limit)
  echo "AWS identity: $(aws sts get-caller-identity --query Arn --output text)"
  local acct; acct=$(aws sts get-caller-identity --query Account --output text)
  aws budgets describe-budgets --account-id "$acct" --output text \
    --query 'Budgets[].[BudgetName,BudgetLimit.Amount,CalculatedSpend.ActualSpend.Amount]' 2>/dev/null \
    | sed 's/^/budget (name, limit, spent): /' || echo "budgets: not readable with this profile; check headroom in the console"
}

ec2_ami() {           # ec2_ami <region>: latest Canonical Ubuntu 24.04 amd64 (task images are amd64-only)
  aws ec2 describe-images --region "$1" --owners 099720109477 --output text \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId'
}

ec2_sg() {            # ec2_sg <region> <name> <description>: the group's id, created if missing, SSH open to this laptop
  local sg; sg=$(aws ec2 describe-security-groups --region "$1" --filters "Name=group-name,Values=$2" \
    --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$sg" = None ]; then
    sg=$(aws ec2 create-security-group --region "$1" --group-name "$2" --description "$3" --query GroupId --output text)
  fi
  ec2_allow_ssh "$1" "$sg" "$(curl -s https://checkip.amazonaws.com)"
  echo "$sg"
}

ec2_allow_ssh() {     # ec2_allow_ssh <region> <sg> <ip>: SSH from that one address (no-op if already allowed)
  aws ec2 authorize-security-group-ingress --region "$1" --group-id "$2" --protocol tcp --port 22 \
    --cidr "$3/32" >/dev/null 2>&1 || true
}

ec2_wait_ssh() {      # ec2_wait_ssh <key> <ip>: until sshd answers
  until ssh -i "$1" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes "ubuntu@$2" true 2>/dev/null; do
    sleep 5
  done
}
