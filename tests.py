
'''

num = int(input())

# # num = 5321
# num = 7820
# # num = 9663

corecteness_flag = True
last_digit = num % 10
# print("last_digit before the cycle = ", last_digit)
pre_last_digit = None

while num != 0:
    num = num // 10
    if num != 0:
        # num = num // 10
        # print("while num = ", num)
        pre_last_digit = num % 10
        # print("pre_last_digit after while = ", pre_last_digit)
        # print("last_digit before IF =", last_digit)
        if last_digit > pre_last_digit:
            corecteness_flag = False
        last_digit = num % 10
        #  print("last_digit before IF =", last_digit)
        # print("last_digit after if = ", last_digit)
    # else: 


if corecteness_flag == True:
    print('YES')
else:
    print('NO')
'''
#####################################################################
'''
# num = int(input())

num = 256
# num = 13
# num = 4683
# num = 1876438942
# num = 18652430
# num = 8500

not_chyotnta_digit_flag = True
i = 0

while not_chyotnta_digit_flag:
    i += 1
    digit = num % 10
    if digit % 2 == 0:
        
        print(i, "-я четная цифра равна", digit)
        
    if digit == 0:
        
        not_chyotnta_digit_flag = False
    if digit % 2 != 0:
        print("Четных цифр в числе нет")
        not_chyotnta_digit_flag = False
    num = num // 10

'''

#####################################################################

for i in range(1, 101):
    if i == 7 or i == 17 or i == 29 or i == 78:
        continue  # переходим на следующую итерацию
    print(i)

print("#" * 80)

for i in range(10):
    print(i, end='*')