"""
Seed script to populate the database with default resume templates.
Run this once after migration: python seed_templates.py
"""

from database import SessionLocal, engine
from models import Template, Base
from datetime import datetime

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

def seed_templates():
    db = SessionLocal()
    
    try:
        # Check if templates already exist
        existing = db.query(Template).count()
        if existing > 0:
            print(f"✅ Templates already seeded ({existing} templates found)")
            return
        
        templates = [
            {
                "name": "Professional - Modern",
                "description": "Clean, modern template with clear section headers. Best for tech and corporate roles.",
                "markdown_content": """# [Your Name]
**[Your Title/Role]** | [City, State] | [Email] | [Phone] | [LinkedIn]

## Professional Summary
[2-3 sentences describing your expertise, key skills, and career goals]

## Technical Skills
**Programming Languages:** Python, JavaScript, Java  
**Frameworks & Tools:** React, Node.js, Docker, AWS  
**Databases:** PostgreSQL, MongoDB, Redis

## Professional Experience

### [Company Name] | [Job Title] | [Start Date] - [End Date]
- Achieved [quantifiable result] by implementing [technology/method]
- Led team of [X] engineers to deliver [project/product]
- Reduced [metric] by [percentage] through [action]

### [Company Name] | [Job Title] | [Start Date] - [End Date]
- Developed [feature/system] using [technologies]
- Collaborated with [teams] to [accomplish goal]
- Optimized [process] resulting in [improvement]

## Education
**[Degree]** in [Field] | [University Name] | [Graduation Year]  
GPA: [X.XX] | Relevant Coursework: [Course 1, Course 2, Course 3]

## Projects
**[Project Name]** | [Technologies Used]  
[Brief description of what the project does and your role]

## Certifications
- [Certification Name] - [Issuing Organization] ([Year])
- [Certification Name] - [Issuing Organization] ([Year])
""",
                "is_default": 1
            },
            {
                "name": "Executive - Senior",
                "description": "Leadership-focused template for senior roles and executives.",
                "markdown_content": """# [Your Name]
**[Executive Title]** | [Location] | [Email] | [Phone] | [LinkedIn]

## Executive Summary
[3-4 sentences highlighting leadership experience, strategic achievements, and executive capabilities]

## Core Competencies
**Leadership:** Team Building, Strategic Planning, Change Management  
**Business:** P&L Management, Budget Planning, Stakeholder Engagement  
**Technical:** [Industry-specific expertise]

## Professional Experience

### [Company Name] | [C-Level / VP Title] | [Years]
**Key Achievements:**
- Drove [$XXM] revenue growth through [strategic initiative]
- Led transformation of [department/division] with [X]+ person team
- Established partnerships with [key stakeholders/clients]
- Implemented [strategy] reducing costs by [percentage]

### [Company Name] | [Director / Senior Manager Title] | [Years]
**Key Achievements:**
- Scaled [product/service] from [X] to [Y] users/revenue
- Built and managed team of [X]+ professionals
- Directed [budget amount] annual budget

## Education
**[Advanced Degree]** | [University] | [Year]  
**[Bachelor's Degree]** | [University] | [Year]

## Board Memberships & Advisory Roles
- [Organization Name] - [Role] ([Years])

## Speaking & Publications
- [Conference/Publication Name] - [Topic] ([Year])
""",
                "is_default": 0
            },
            {
                "name": "Creative - Designer",
                "description": "Visual and creative template for designers, artists, and creative professionals.",
                "markdown_content": """# [Your Name]
### [Creative Title] | [Portfolio Link] | [Email]

## About Me
[Brief creative statement highlighting your design philosophy and specialties]

## Skills & Tools
**Design:** UI/UX Design, Brand Identity, Typography, Color Theory  
**Software:** Figma, Adobe Creative Suite, Sketch, InVision  
**Development:** HTML, CSS, JavaScript (basic), React (basic)

## Professional Experience

### [Company/Agency] | [Design Role] | [Dates]
**Projects:**
- **[Project Name]:** Redesigned [product] resulting in [metric improvement]
- **[Project Name]:** Created brand identity for [client] including logo, style guide, and marketing materials
- **[Project Name]:** Led user research and designed [feature] increasing [engagement metric] by [X]%

### [Company/Agency] | [Design Role] | [Dates]
**Projects:**
- Collaborated with product team to design [X] features
- Conducted user interviews and usability testing
- Maintained design system with [X] components

## Featured Projects
**[Project Name]** | [Year]  
[What it is, your role, technologies/tools used, and impact]

**[Project Name]** | [Year]  
[What it is, your role, technologies/tools used, and impact]

## Education & Certifications
**[Degree]** in [Field] | [School] | [Year]  
**Google UX Design Certificate** | [Year]

## Awards & Recognition
- [Award Name] - [Organization] ([Year])
""",
                "is_default": 0
            }
        ]
        
        for template_data in templates:
            template = Template(**template_data)
            db.add(template)
        
        db.commit()
        print(f"✅ Successfully seeded {len(templates)} templates!")
        
        # Print template names
        for t in templates:
            print(f"   - {t['name']}")
        
    except Exception as e:
        print(f"❌ Error seeding templates: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🌱 Seeding templates...")
    seed_templates()
