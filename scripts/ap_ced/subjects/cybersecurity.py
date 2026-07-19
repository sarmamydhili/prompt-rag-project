"""AP Cybersecurity CED subject configuration."""

from __future__ import annotations

from collections import OrderedDict

from ..config import SubjectConfig, UnitMeta

CYBERSECURITY = SubjectConfig(
    subject="AP Cybersecurity",
    slug="cybersecurity",
    footer_prefix="AP Cybersecurity",
    first_topic_id="1.1",
    last_topic_id="5.6",
    unit_header_names=[
        "Introduction to Security",
        "Securing Spaces",
        "Securing Networks",
        "Securing Devices",
        "Securing Applications and Data",
    ],
    skill_categories=[
        {"id": 1, "name": "Analyze Risk", "color": "#B12D89"},
        {"id": 2, "name": "Mitigate Risk", "color": "#006EAF"},
        {"id": 3, "name": "Detect Attacks", "color": "#C98E71"},
        {"id": 4, "name": "Collaborate", "color": "#E9A612"},
    ],
    units={
        1: UnitMeta("Introduction to Security", 10),
        2: UnitMeta("Securing Spaces", 21),
        3: UnitMeta("Securing Networks", 26),
        4: UnitMeta("Securing Devices", 23),
        5: UnitMeta("Securing Applications and Data", 30),
    },
    topic_titles=OrderedDict(
        [
            ("1.1", "Understanding Social Engineering"),
            ("1.2", "Suspicious Website Logins"),
            ("1.3", "Best Practices for Public Networks"),
            ("1.4", "AI-Based Cybersecurity Attacks"),
            ("1.5", "Leveraging AI in Cyber Defense"),
            ("2.1", "Cyber Foundations"),
            ("2.2", "Physical Vulnerabilities and Attacks"),
            ("2.3", "Protecting Physical Spaces"),
            ("2.4", "Detecting Physical Attacks"),
            ("3.1", "Network Vulnerabilities and Attacks"),
            ("3.2", "Protecting Networks: Managerial Controls and Wireless Security"),
            ("3.3", "Protecting Networks: Segmentation"),
            ("3.4", "Protecting Networks: Firewalls"),
            ("3.5", "Detecting Network Attacks"),
            ("4.1", "Device Vulnerabilities and Attacks"),
            ("4.2", "Authentication"),
            ("4.3", "Protecting Devices"),
            ("4.4", "Detecting Attacks on Devices"),
            ("5.1", "Application and Data Vulnerabilities and Attacks"),
            ("5.2", "Protecting Applications and Data: Managerial Controls and Access Controls"),
            ("5.3", "Protecting Stored Data with Cryptography"),
            ("5.4", "Asymmetric Cryptography"),
            ("5.5", "Protecting Applications"),
            ("5.6", "Detecting Attacks on Data and Applications"),
        ]
    ),
    manual_los={
        "3.4.A": "Configure a firewall to manage the flow of network traffic.",
        "3.4.B": "Identify types of network-based firewalls.",
        "3.4.C": "Explain how a firewall uses an access control list to allow or deny traffic entering or leaving a network.",
        "3.4.D": "Determine the effective placement of firewalls in a network.",
    },
    scenario_title_overrides={
        "3C": "Protecting a Network on a Naval Submarine",
    },
)
