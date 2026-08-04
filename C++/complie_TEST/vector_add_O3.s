	.file	"vector_add.cpp"
	.text
	.p2align 4
	.globl	_Z9vectorAddPiS_m
	.type	_Z9vectorAddPiS_m, @function
_Z9vectorAddPiS_m:
.LFB14:
	.cfi_startproc
	endbr64
	movq	%rsi, %rax
	movq	%rdi, %rcx
	movq	%rdx, %rsi
	testq	%rdx, %rdx
	je	.L1
	leaq	-1(%rdx), %rdi
	cmpq	$3, %rdi
	jbe	.L19
	leaq	4(%rcx), %r8
	movq	%rax, %r9
	xorl	%edx, %edx
	subq	%r8, %r9
	cmpq	$8, %r9
	ja	.L43
	.p2align 4,,10
	.p2align 3
.L7:
	movl	$1, (%rcx,%rdx,4)
	movl	%edx, (%rax,%rdx,4)
	addq	$1, %rdx
	cmpq	%rdx, %rsi
	jne	.L7
	cmpq	$1, %rsi
	jne	.L44
	movl	(%rcx), %edx
	addl	%edx, (%rax)
.L1:
	ret
	.p2align 4,,10
	.p2align 3
.L43:
	movq	%rsi, %rdi
	movdqa	.LC0(%rip), %xmm1
	movdqa	.LC1(%rip), %xmm5
	shrq	$2, %rdi
	movdqa	.LC2(%rip), %xmm4
	movdqa	.LC3(%rip), %xmm3
	salq	$4, %rdi
	.p2align 4,,10
	.p2align 3
.L4:
	movdqa	%xmm1, %xmm0
	movups	%xmm4, (%rcx,%rdx)
	paddq	%xmm5, %xmm1
	movdqa	%xmm0, %xmm2
	paddq	%xmm3, %xmm2
	shufps	$136, %xmm2, %xmm0
	movups	%xmm0, (%rax,%rdx)
	addq	$16, %rdx
	cmpq	%rdx, %rdi
	jne	.L4
	movq	%rsi, %rdx
	andq	$-4, %rdx
	testb	$3, %sil
	je	.L9
	leaq	1(%rdx), %r9
	movl	$1, (%rcx,%rdx,4)
	leaq	0(,%rdx,4), %rdi
	movl	%edx, (%rax,%rdx,4)
	cmpq	%rsi, %r9
	jnb	.L6
	addq	$2, %rdx
	movl	$1, 4(%rcx,%rdi)
	movl	%r9d, 4(%rax,%rdi)
	cmpq	%rsi, %rdx
	jnb	.L6
	movl	$1, 8(%rcx,%rdi)
	movl	%edx, 8(%rax,%rdi)
.L9:
	movq	%rsi, %rdi
	xorl	%edx, %edx
	shrq	$2, %rdi
	salq	$4, %rdi
	.p2align 4,,10
	.p2align 3
.L13:
	movdqu	(%rax,%rdx), %xmm0
	movdqu	(%rcx,%rdx), %xmm6
	paddd	%xmm6, %xmm0
	movups	%xmm0, (%rax,%rdx)
	addq	$16, %rdx
	cmpq	%rdx, %rdi
	jne	.L13
	testb	$3, %sil
	je	.L1
	movq	%rsi, %rdx
	andq	$-4, %rdx
	subq	%rdx, %rsi
	cmpq	$1, %rsi
	je	.L15
.L12:
	leaq	(%rax,%rdx,4), %rdi
	movq	(%rcx,%rdx,4), %xmm0
	movq	(%rdi), %xmm1
	paddd	%xmm1, %xmm0
	movq	%xmm0, (%rdi)
	testb	$1, %sil
	je	.L1
	andq	$-2, %rsi
	addq	%rsi, %rdx
.L15:
	movl	(%rcx,%rdx,4), %ecx
	addl	%ecx, (%rax,%rdx,4)
	ret
	.p2align 4,,10
	.p2align 3
.L44:
	leaq	4(%rcx), %r8
	movq	%rax, %rdx
	subq	%r8, %rdx
	cmpq	$8, %rdx
	ja	.L45
.L10:
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L17:
	movl	(%rcx,%rdx,4), %edi
	addl	%edi, (%rax,%rdx,4)
	addq	$1, %rdx
	cmpq	%rdx, %rsi
	jne	.L17
	ret
	.p2align 4,,10
	.p2align 3
.L19:
	xorl	%edx, %edx
	jmp	.L7
.L45:
	xorl	%edx, %edx
	cmpq	$2, %rdi
	jbe	.L12
	jmp	.L9
.L6:
	movq	%rax, %rdx
	subq	%r8, %rdx
	cmpq	$8, %rdx
	jbe	.L10
	jmp	.L9
	.cfi_endproc
.LFE14:
	.size	_Z9vectorAddPiS_m, .-_Z9vectorAddPiS_m
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC0:
	.quad	0
	.quad	1
	.align 16
.LC1:
	.quad	4
	.quad	4
	.align 16
.LC2:
	.long	1
	.long	1
	.long	1
	.long	1
	.align 16
.LC3:
	.quad	2
	.quad	2
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
