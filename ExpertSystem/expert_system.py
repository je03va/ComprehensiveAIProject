"""
Network Security Configuration Advisor

Expert System assignment (moved here from csc 470/expertSystemAssignment.ipynb so the
integrated hub can import it). Run on its own with:  python expert_system.py
"""
import json
import os

KB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network_security_kb.json")

DEFAULT_KB = {
    "facts": {
        "default_credentials": "Are any devices still using default/factory login credentials?",
        "no_firewall": "Is there no firewall (or an unconfigured firewall) protecting the network?",
        "open_ports": "Are there open ports exposed directly to the internet?",
        "unencrypted_traffic": "Is sensitive traffic sent without encryption (e.g. plain HTTP, FTP)?",
        "outdated_firmware": "Is any router/switch/firewall firmware out of date?",
        "weak_wifi_encryption": "Is the wifi using weak or no encryption (open, WEP, or WPA)?",
        "no_network_segmentation": "Are all devices on a single flat network with no segmentation/VLANs?",
        "unrestricted_remote_access": "Is remote access (RDP/SSH/VPN) allowed from any IP with no restrictions?",
        "no_logging_monitoring": "Is there no logging or monitoring of network/security events?",
        "guest_network_shared": "Does the guest wifi share the same network as internal systems?",
    },
    "rules": {
        "Default Credentials Exposure": {
            "facts": ["default_credentials"],
            "severity": "Critical",
            "confidence": 0.95,
            "recommendation": [
                "Change all default usernames/passwords immediately.",
                "Enforce a password policy for all network devices.",
                "Audit devices periodically for default credentials.",
            ],
        },
        "Unrestricted Inbound Access": {
            "facts": ["no_firewall", "open_ports"],
            "severity": "High",
            "confidence": 0.85,
            "recommendation": [
                "Deploy and configure a firewall with a default-deny inbound policy.",
                "Close any ports not explicitly required.",
                "Restrict remaining open ports to known source IPs where possible.",
            ],
        },
        "Data in Transit Exposure": {
            "facts": ["unencrypted_traffic"],
            "severity": "High",
            "confidence": 0.8,
            "recommendation": [
                "Enforce TLS/HTTPS and SFTP/SSH for all sensitive traffic.",
                "Disable legacy unencrypted protocols where possible.",
            ],
        },
        "Known Vulnerability Exposure": {
            "facts": ["outdated_firmware"],
            "severity": "Critical",
            "confidence": 0.9,
            "recommendation": [
                "Patch/update firmware on all network devices.",
                "Subscribe to vendor security advisories.",
                "Establish a regular patch-management schedule.",
            ],
        },
        "Wireless Eavesdropping Risk": {
            "facts": ["weak_wifi_encryption"],
            "severity": "Medium",
            "confidence": 0.75,
            "recommendation": [
                "Switch to WPA3 (or WPA2-AES minimum).",
                "Rotate wifi passwords and disable WPS.",
            ],
        },
        "Lateral Movement Risk": {
            "facts": ["no_network_segmentation"],
            "severity": "High",
            "confidence": 0.7,
            "recommendation": [
                "Segment the network with VLANs by function/sensitivity.",
                "Apply inter-VLAN access control rules.",
            ],
        },
        "Remote Exploitation Risk": {
            "facts": ["unrestricted_remote_access"],
            "severity": "Critical",
            "confidence": 0.9,
            "recommendation": [
                "Restrict remote access to a VPN with MFA.",
                "Whitelist source IPs where feasible.",
                "Disable direct RDP/SSH exposure to the internet.",
            ],
        },
        "Blind Spot in Detection": {
            "facts": ["no_logging_monitoring"],
            "severity": "Medium",
            "confidence": 0.65,
            "recommendation": [
                "Enable logging on firewalls, routers, and critical servers.",
                "Centralize logs and set alerting for anomalies.",
            ],
        },
        "Guest Network Compromise Risk": {
            "facts": ["guest_network_shared"],
            "severity": "Medium",
            "confidence": 0.7,
            "recommendation": [
                "Put the guest network on its own isolated VLAN.",
                "Block guest-to-internal traffic entirely.",
            ],
        },
    },
}


class NetworkSecurityAdvisor:

    def __init__(self):
        kb = self.load_knowledge_base()
        self.facts_catalog = kb["facts"]
        self.rules = kb["rules"]
        self.reported_facts = set()
        self.results = []

    def load_knowledge_base(self):
        if os.path.exists(KB_FILE):
            with open(KB_FILE, "r") as f:
                return json.load(f)
        with open(KB_FILE, "w") as f:
            json.dump(DEFAULT_KB, f, indent=2)
        return json.loads(json.dumps(DEFAULT_KB))

    def save_knowledge_base(self):
        with open(KB_FILE, "w") as f:
            json.dump({"facts": self.facts_catalog, "rules": self.rules}, f, indent=2)

    def user_interface(self):
        print("\nWelcome to the Network Security Configuration Advisor.")
        menu = """
        0. Quit
        1. KNOWLEDGE ACQUISITION (add a new rule)
        2. ASSESS A NETWORK CONFIGURATION
        """
        while True:
            print(menu)
            try:
                choice = int(input("Choose an option: "))
                if choice == 0:
                    print("Goodbye!")
                    break
                elif choice == 1:
                    self.acquire_knowledge()
                elif choice == 2:
                    self.assess()
                else:
                    print("Invalid option, try again.")
            except ValueError:
                print("Please enter a number.")

    # ------------------------------------------------------------------
    # Knowledge acquisition
    # ------------------------------------------------------------------
    def acquire_knowledge(self):
        print("\nKnowledge acquisition (security engineer access).")
        name = input("New vulnerability/rule name: ").strip()
        if not name or name in self.rules:
            print("Invalid or duplicate rule name. Cancelling.")
            return

        print("Enter the trigger facts for this rule, one per line, blank to finish.")
        print("(Existing facts: " + ", ".join(self.facts_catalog.keys()) + ")")
        facts = []
        while True:
            key = input("  Fact key (short_snake_case), or blank to stop: ").strip().lower()
            if not key:
                break
            if key not in self.facts_catalog:
                question = input(f"    '{key}' is new. Enter the yes/no question to ask for it: ").strip()
                self.facts_catalog[key] = question or f"Is '{key}' true?"
            facts.append(key)

        if not facts:
            print("No facts entered. Cancelling.")
            return

        severity = ""
        while severity not in ("low", "medium", "high", "critical"):
            severity = input("Severity (Low/Medium/High/Critical): ").strip().lower()

        while True:
            try:
                confidence = float(input("Confidence this rule indicates real risk (0.1-1.0): "))
                if 0.1 <= confidence <= 1.0:
                    break
            except ValueError:
                pass
            print("Enter a number between 0.1 and 1.0.")

        print("Enter hardening/recommendation steps, one per line, blank to finish.")
        steps = []
        while True:
            step = input("  Step: ").strip()
            if not step:
                break
            steps.append(step)

        self.rules[name] = {
            "facts": facts,
            "severity": severity.capitalize(),
            "confidence": confidence,
            "recommendation": steps or ["No steps recorded yet."],
        }
        self.save_knowledge_base()
        print(f"'{name}' added and saved.")

    # ------------------------------------------------------------------
    # Interview + forward-chaining inference
    # ------------------------------------------------------------------
    def assess(self):
        print("\nAnswer yes/no about the network configuration being assessed.")
        reported = set()
        for fact, question in self.facts_catalog.items():
            ans = input(f"  {question} (y/n): ").strip().lower()
            if ans.startswith("y"):
                reported.add(fact)
        self.reported_facts = reported
        self.infer()
        self.explain()

    def infer(self):
        severity_rank = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
        fired = []
        for vuln, rule in self.rules.items():
            required = set(rule["facts"])
            if required.issubset(self.reported_facts):
                fired.append((vuln, rule))
        fired.sort(key=lambda x: (severity_rank.get(x[1]["severity"], 0), x[1]["confidence"]), reverse=True)
        self.results = fired

    def explain(self):
        if not self.results:
            print("\nNo rules fired - no vulnerabilities flagged from the facts reported.")
            print("(This reflects the current rule base only, not a guarantee of security.)")
            return

        print("\nSECURITY ASSESSMENT RESULTS")
        for vuln, rule in self.results:
            print(f"\n>> {vuln}  (severity: {rule['severity']}, confidence: {rule['confidence']*100:.0f}%)")
            print(f"   Why: triggered by -> {rule['facts']}")
            print("   Recommended hardening steps:")
            for i, step in enumerate(rule["recommendation"], 1):
                print(f"     {i}. {step}")


if __name__ == "__main__":
    handle = NetworkSecurityAdvisor()
    handle.user_interface()
