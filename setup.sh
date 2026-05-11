#!/bin/bash
# =============================================================================
# setup.sh — Guardian AI full setup script
# Run this ONCE on your Ubuntu machine where Wazuh is installed.
# It does everything: moves files, configures Wazuh, trains, starts the AI.
#
# Usage:
#   chmod +x setup.sh
#   sudo bash setup.sh
# =============================================================================

set -e  # stop on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }
info() { echo -e "${BLUE}[→]${NC} $1"; }

echo ""
echo "════════════════════════════════════════════════════════"
echo "   GUARDIAN AI — Full Setup"
echo "   Wazuh 4.x + Ubuntu"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Check root ────────────────────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    err "Please run as root: sudo bash setup.sh"
fi

# ── Check Wazuh is installed ──────────────────────────────────────────────────
if [ ! -f /var/ossec/bin/wazuh-control ]; then
    err "Wazuh is not installed at /var/ossec. Install Wazuh first."
fi
log "Wazuh installation found"

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    err "python3 not found. Install with: apt install python3"
fi
log "Python3 found: $(python3 --version)"

# ── Install Python dependencies ───────────────────────────────────────────────
info "Installing Python dependencies..."

# Ubuntu 22.04 has old pip — use apt packages first, then pip without flags
apt install -y python3-sklearn python3-joblib python3-pandas python3-numpy 2>/dev/null || true

# Then install via pip for any missing ones — try multiple methods
python3 -m pip install scikit-learn joblib pandas numpy 2>/dev/null || \
python3 -m pip install scikit-learn joblib pandas numpy --user 2>/dev/null || \
pip3 install scikit-learn joblib pandas numpy 2>/dev/null || true

# Verify installation worked
python3 -c "import sklearn, joblib, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    # Last resort: install newer pip first then install packages
    python3 -m pip install --upgrade pip 2>/dev/null || true
    python3 -m pip install scikit-learn joblib pandas numpy --user
fi

log "Python dependencies installed"

# Add local bin to PATH for this session
export PATH="$PATH:/root/.local/bin:/home/$SUDO_USER/.local/bin"

# Make sure python can find the packages when running as root
SITE_PACKAGES=$(python3 -c "import site; print(site.getusersitepackages())" 2>/dev/null || echo "")
USER_SITE=$(python3 -m site --user-site 2>/dev/null || echo "")
log "Python packages location: $USER_SITE"

# ── Create /guardian directory ────────────────────────────────────────────────
info "Creating /guardian directory..."
mkdir -p /guardian
log "/guardian directory ready"

# ── Copy AI files to /guardian ────────────────────────────────────────────────
info "Copying AI files to /guardian..."

# Find where the AI files are (same directory as this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AI_FILES=(
    "feature_extractor.py"
    "temporal_builder.py"
    "attack_correlator.py"
    "model.py"
    "predictor.py"
    "guardian_live.py"
    "train_guardian.py"
)

for f in "${AI_FILES[@]}"; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" /guardian/
        log "Copied $f"
    else
        err "Missing file: $f — make sure all AI files are in the same folder as setup.sh"
    fi
done

# ── Install Wazuh custom rules ────────────────────────────────────────────────
info "Installing Guardian AI rules into Wazuh..."
cp "$SCRIPT_DIR/guardian_rules.xml" /var/ossec/etc/rules/guardian_rules.xml
chown root:wazuh /var/ossec/etc/rules/guardian_rules.xml
chmod 660 /var/ossec/etc/rules/guardian_rules.xml
log "Guardian rules installed"

# ── Configure Wazuh to read AI predictions ────────────────────────────────────
info "Configuring Wazuh to read AI prediction output..."

OSSEC_CONF="/var/ossec/etc/ossec.conf"

# Check if already configured
if grep -q "guardian_predictions" "$OSSEC_CONF"; then
    warn "Guardian localfile already in ossec.conf — skipping"
else
    # Add localfile block before </ossec_config>
    LOCALFILE_BLOCK='
  <!-- Guardian AI predictions feed -->
  <localfile>
    <log_format>json</log_format>
    <location>/var/ossec/logs/alerts/guardian_predictions.json</location>
    <label key="source">guardian-ai</label>
  </localfile>'

    # Insert before the last </ossec_config>
    sed -i "s|</ossec_config>|${LOCALFILE_BLOCK}\n</ossec_config>|" "$OSSEC_CONF"
    log "Wazuh ossec.conf updated"
fi

# ── Create the predictions file with correct permissions ──────────────────────
touch /var/ossec/logs/alerts/guardian_predictions.json
chown root:wazuh /var/ossec/logs/alerts/guardian_predictions.json
chmod 664 /var/ossec/logs/alerts/guardian_predictions.json
log "Predictions file created"

# ── Train the AI model ────────────────────────────────────────────────────────
info "Training the AI model (this takes ~30 seconds)..."
python3 /guardian/train_guardian.py
log "AI model trained and saved to /guardian/guardian_model.joblib"

# ── Install systemd service ───────────────────────────────────────────────────
info "Installing Guardian as a system service..."
cp "$SCRIPT_DIR/guardian.service" /etc/systemd/system/guardian.service
systemctl daemon-reload
systemctl enable guardian
log "Guardian service installed and enabled at startup"

# ── Restart Wazuh to pick up new rules ───────────────────────────────────────
info "Restarting Wazuh manager to load new rules..."
systemctl restart wazuh-manager
sleep 5
log "Wazuh manager restarted"

# ── Start the Guardian AI service ────────────────────────────────────────────
info "Starting Guardian AI service..."
systemctl start guardian
sleep 3

# Check it started
if systemctl is-active --quiet guardian; then
    log "Guardian AI service is RUNNING"
else
    err "Guardian service failed to start. Check: journalctl -u guardian -n 50"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo -e "   ${GREEN}GUARDIAN AI IS NOW ACTIVE${NC}"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  Status    : sudo systemctl status guardian"
echo "  Live logs : sudo journalctl -u guardian -f"
echo "  AI log    : sudo tail -f /var/ossec/logs/guardian_ai.log"
echo "  Predictions: sudo tail -f /var/ossec/logs/alerts/guardian_predictions.json"
echo ""
echo "  Dashboard : open your Wazuh dashboard → Security Events"
echo "              filter by: rule.groups: guardian_ai"
echo ""
echo "  The AI runs every 5 minutes automatically."
echo "  It will also start automatically when Ubuntu boots."
echo ""
