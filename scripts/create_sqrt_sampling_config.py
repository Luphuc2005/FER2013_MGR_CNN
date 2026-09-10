with open('config_rafdb_siglip2_semantic_stable_v5_combined_ultimate.yaml', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'rafdb_siglip2_semantic_stable_v5_combined_ultimate' in line:
        line = line.replace('rafdb_siglip2_semantic_stable_v5_combined_ultimate', 'rafdb_siglip2_semantic_stable_v5_combined_ultimate_sqrt_sampling')
    if 'note: "RAF-DB v5 Combined Ultimate:' in line:
        line = '  note: "RAF-DB v5 Combined Ultimate + Square-Root Sampling (gamma=0.5) to balance minority classes (Fear 2.3%->6.2%, Disgust 5.8%->9.9%). Standard Train=11043, Val=1228, Test=3068 split."\n'
    new_lines.append(line)
    if '  cache: false' in line:
        new_lines.append('  sampling_strategy: "square_root"\n')
        new_lines.append('  sampling_gamma: 0.5\n')

with open('config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_sqrt_sampling.yaml', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('[OK] Created config_rafdb_siglip2_semantic_stable_v5_combined_ultimate_sqrt_sampling.yaml successfully!')