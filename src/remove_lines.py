# with open("spec-pip.txt") as f:
#     text = f.read()

# # new = text.replace("  - ", "")
# new = text.replace("      - ", "")

# print(new)

# with open("spec-pip-new.txt", "w") as f:
#     f.write(new)



with open("spec-conda-scrubbed.txt") as f:
    text = f.readlines()

new = ""

for line in text:
    new = new + "  - " + line

with open("spec-pip-scrubbbed-lines.txt", "w") as f:
    f.write(new)