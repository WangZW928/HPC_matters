	.file	"vector_add.cpp"
	.text
	.p2align 4
	.globl	_Z9vectorAddPiS_m
	.type	_Z9vectorAddPiS_m, @function
_Z9vectorAddPiS_m:
.LFB14:
	.cfi_startproc
	endbr64
	testq	%rdx, %rdx
	je	.L1
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L3:
	movl	$1, (%rdi,%rax,4)
	movq	%rax, %rcx
	movl	%eax, (%rsi,%rax,4)
	leaq	1(%rax), %rax
	cmpq	%rax, %rdx
	jne	.L3
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L4:
	movl	(%rdi,%rax,4), %edx
	addl	%edx, (%rsi,%rax,4)
	movq	%rax, %rdx
	addq	$1, %rax
	cmpq	%rdx, %rcx
	jne	.L4
.L1:
	ret
	.cfi_endproc
.LFE14:
	.size	_Z9vectorAddPiS_m, .-_Z9vectorAddPiS_m
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
