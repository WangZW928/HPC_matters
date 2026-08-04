	.file	"main.cpp"
	.text
	.section	.text._ZSt23__is_constant_evaluatedv,"axG",@progbits,_ZSt23__is_constant_evaluatedv,comdat
	.weak	_ZSt23__is_constant_evaluatedv
	.type	_ZSt23__is_constant_evaluatedv, @function
_ZSt23__is_constant_evaluatedv:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$0, %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	_ZSt23__is_constant_evaluatedv, .-_ZSt23__is_constant_evaluatedv
	.section	.rodata
.LC0:
	.string	"b[0] = %d\n"
.LC1:
	.string	"b[63] = %d\n"
.LC2:
	.string	"checksum = %lld\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1575:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA1575
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$64, -32(%rbp)
	leaq	-64(%rbp), %rax
	movl	$64, %esi
	movq	%rax, %rdi
.LEHB0:
	call	_ZSt11make_uniqueIA_iENSt8__detail9_MakeUniqIT_E7__arrayEm
.LEHE0:
	leaq	-56(%rbp), %rax
	movl	$64, %esi
	movq	%rax, %rdi
.LEHB1:
	call	_ZSt11make_uniqueIA_iENSt8__detail9_MakeUniqIT_E7__arrayEm
.LEHE1:
	leaq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv
	movq	%rax, %rbx
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv
	movl	$64, %edx
	movq	%rbx, %rsi
	movq	%rax, %rdi
.LEHB2:
	call	_Z9vectorAddPiS_m@PLT
	movq	$0, -48(%rbp)
	movq	$0, -40(%rbp)
	jmp	.L4
.L5:
	movq	-40(%rbp), %rdx
	leaq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EEixEm
	movl	(%rax), %eax
	cltq
	addq	%rax, -48(%rbp)
	addq	$1, -40(%rbp)
.L4:
	cmpq	$63, -40(%rbp)
	jbe	.L5
	leaq	-56(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EEixEm
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-56(%rbp), %rax
	movl	$63, %esi
	movq	%rax, %rdi
	call	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EEixEm
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
.LEHE2:
	cmpq	$2080, -48(%rbp)
	setne	%al
	movzbl	%al, %ebx
	leaq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED1Ev
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED1Ev
	movl	%ebx, %eax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L10
	jmp	.L13
.L12:
	endbr64
	movq	%rax, %rbx
	leaq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED1Ev
	jmp	.L8
.L11:
	endbr64
	movq	%rax, %rbx
.L8:
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED1Ev
	movq	%rbx, %rax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L9
	call	__stack_chk_fail@PLT
.L9:
	movq	%rax, %rdi
.LEHB3:
	call	_Unwind_Resume@PLT
.LEHE3:
.L13:
	call	__stack_chk_fail@PLT
.L10:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1575:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA1575:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE1575-.LLSDACSB1575
.LLSDACSB1575:
	.uleb128 .LEHB0-.LFB1575
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB1575
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L11-.LFB1575
	.uleb128 0
	.uleb128 .LEHB2-.LFB1575
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L12-.LFB1575
	.uleb128 0
	.uleb128 .LEHB3-.LFB1575
	.uleb128 .LEHE3-.LEHB3
	.uleb128 0
	.uleb128 0
.LLSDACSE1575:
	.text
	.size	main, .-main
	.section	.text._ZSt11make_uniqueIA_iENSt8__detail9_MakeUniqIT_E7__arrayEm,"axG",@progbits,_ZSt11make_uniqueIA_iENSt8__detail9_MakeUniqIT_E7__arrayEm,comdat
	.weak	_ZSt11make_uniqueIA_iENSt8__detail9_MakeUniqIT_E7__arrayEm
	.type	_ZSt11make_uniqueIA_iENSt8__detail9_MakeUniqIT_E7__arrayEm, @function
_ZSt11make_uniqueIA_iENSt8__detail9_MakeUniqIT_E7__arrayEm:
.LFB1592:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	-32(%rbp), %rbx
	movabsq	$2305843009213693950, %rax
	cmpq	%rbx, %rax
	jb	.L15
	leaq	0(,%rbx,4), %rax
	movq	%rax, %rdi
	call	_Znam@PLT
	movq	%rax, %rcx
	movq	%rcx, %rdx
	leaq	-1(%rbx), %rax
	jmp	.L17
.L15:
	call	__cxa_throw_bad_array_new_length@PLT
.L18:
	movl	$0, (%rdx)
	subq	$1, %rax
	addq	$4, %rdx
.L17:
	testq	%rax, %rax
	jns	.L18
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC1IPiS2_vbEET_
	movq	-24(%rbp), %rax
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1592:
	.size	_ZSt11make_uniqueIA_iENSt8__detail9_MakeUniqIT_E7__arrayEm, .-_ZSt11make_uniqueIA_iENSt8__detail9_MakeUniqIT_E7__arrayEm
	.section	.text._ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED2Ev,"axG",@progbits,_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED5Ev,comdat
	.align 2
	.weak	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED2Ev
	.type	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED2Ev, @function
_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED2Ev:
.LFB1594:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L21
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EE11get_deleterEv
	movq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	_ZNKSt14default_deleteIA_iEclIiEENSt9enable_ifIXsrSt14is_convertibleIPA_T_PS0_E5valueEvE4typeEPS5_
.L21:
	movq	-8(%rbp), %rax
	movq	$0, (%rax)
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1594:
	.size	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED2Ev, .-_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED2Ev
	.weak	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED1Ev
	.set	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED1Ev,_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EED2Ev
	.section	.text._ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv,"axG",@progbits,_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv,comdat
	.align 2
	.weak	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv
	.type	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv, @function
_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv:
.LFB1596:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1596:
	.size	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv, .-_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv
	.section	.text._ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EEixEm,"axG",@progbits,_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EEixEm,comdat
	.align 2
	.weak	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EEixEm
	.type	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EEixEm, @function
_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EEixEm:
.LFB1597:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	call	_ZSt23__is_constant_evaluatedv
	testb	%al, %al
	je	.L25
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv
	testq	%rax, %rax
	jne	.L25
	movl	$1, %eax
	jmp	.L26
.L25:
	movl	$0, %eax
.L26:
	testb	%al, %al
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EE3getEv
	movq	-16(%rbp), %rdx
	salq	$2, %rdx
	addq	%rdx, %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1597:
	.size	_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EEixEm, .-_ZNKSt10unique_ptrIA_iSt14default_deleteIS0_EEixEm
	.section	.text._ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI2St15__uniq_ptr_implIiS2_EEPi,"axG",@progbits,_ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI5St15__uniq_ptr_implIiS2_EEPi,comdat
	.align 2
	.weak	_ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI2St15__uniq_ptr_implIiS2_EEPi
	.type	_ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI2St15__uniq_ptr_implIiS2_EEPi, @function
_ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI2St15__uniq_ptr_implIiS2_EEPi:
.LFB1610:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC2EPi
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1610:
	.size	_ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI2St15__uniq_ptr_implIiS2_EEPi, .-_ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI2St15__uniq_ptr_implIiS2_EEPi
	.weak	_ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI1St15__uniq_ptr_implIiS2_EEPi
	.set	_ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI1St15__uniq_ptr_implIiS2_EEPi,_ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI2St15__uniq_ptr_implIiS2_EEPi
	.section	.text._ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC2IPiS2_vbEET_,"axG",@progbits,_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC5IPiS2_vbEET_,comdat
	.align 2
	.weak	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC2IPiS2_vbEET_
	.type	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC2IPiS2_vbEET_, @function
_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC2IPiS2_vbEET_:
.LFB1612:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt15__uniq_ptr_dataIiSt14default_deleteIA_iELb1ELb1EECI1St15__uniq_ptr_implIiS2_EEPi
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1612:
	.size	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC2IPiS2_vbEET_, .-_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC2IPiS2_vbEET_
	.weak	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC1IPiS2_vbEET_
	.set	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC1IPiS2_vbEET_,_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EEC2IPiS2_vbEET_
	.section	.text._ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv,"axG",@progbits,_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv,comdat
	.align 2
	.weak	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv
	.type	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv, @function
_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv:
.LFB1614:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt3getILm0EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1614:
	.size	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv, .-_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv
	.section	.text._ZNSt10unique_ptrIA_iSt14default_deleteIS0_EE11get_deleterEv,"axG",@progbits,_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EE11get_deleterEv,comdat
	.align 2
	.weak	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EE11get_deleterEv
	.type	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EE11get_deleterEv, @function
_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EE11get_deleterEv:
.LFB1615:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE10_M_deleterEv
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1615:
	.size	_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EE11get_deleterEv, .-_ZNSt10unique_ptrIA_iSt14default_deleteIS0_EE11get_deleterEv
	.section	.text._ZNKSt14default_deleteIA_iEclIiEENSt9enable_ifIXsrSt14is_convertibleIPA_T_PS0_E5valueEvE4typeEPS5_,"axG",@progbits,_ZNKSt14default_deleteIA_iEclIiEENSt9enable_ifIXsrSt14is_convertibleIPA_T_PS0_E5valueEvE4typeEPS5_,comdat
	.align 2
	.weak	_ZNKSt14default_deleteIA_iEclIiEENSt9enable_ifIXsrSt14is_convertibleIPA_T_PS0_E5valueEvE4typeEPS5_
	.type	_ZNKSt14default_deleteIA_iEclIiEENSt9enable_ifIXsrSt14is_convertibleIPA_T_PS0_E5valueEvE4typeEPS5_, @function
_ZNKSt14default_deleteIA_iEclIiEENSt9enable_ifIXsrSt14is_convertibleIPA_T_PS0_E5valueEvE4typeEPS5_:
.LFB1616:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	cmpq	$0, -16(%rbp)
	je	.L37
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	_ZdaPv@PLT
.L37:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1616:
	.size	_ZNKSt14default_deleteIA_iEclIiEENSt9enable_ifIXsrSt14is_convertibleIPA_T_PS0_E5valueEvE4typeEPS5_, .-_ZNKSt14default_deleteIA_iEclIiEENSt9enable_ifIXsrSt14is_convertibleIPA_T_PS0_E5valueEvE4typeEPS5_
	.section	.text._ZNKSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv,"axG",@progbits,_ZNKSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv,comdat
	.align 2
	.weak	_ZNKSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv
	.type	_ZNKSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv, @function
_ZNKSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv:
.LFB1617:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt3getILm0EJPiSt14default_deleteIA_iEEERKNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERKS8_
	movq	(%rax), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1617:
	.size	_ZNKSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv, .-_ZNKSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv
	.section	.text._ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC2EPi,"axG",@progbits,_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC5EPi,comdat
	.align 2
	.weak	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC2EPi
	.type	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC2EPi, @function
_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC2EPi:
.LFB1624:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC1ILb1ELb1EEEv
	movq	-32(%rbp), %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE6_M_ptrEv
	movq	%rbx, (%rax)
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1624:
	.size	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC2EPi, .-_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC2EPi
	.weak	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC1EPi
	.set	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC1EPi,_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEEC2EPi
	.section	.text._ZSt3getILm0EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_,"axG",@progbits,_ZSt3getILm0EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_,comdat
	.weak	_ZSt3getILm0EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_
	.type	_ZSt3getILm0EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_, @function
_ZSt3getILm0EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_:
.LFB1626:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1626:
	.size	_ZSt3getILm0EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_, .-_ZSt3getILm0EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_
	.section	.text._ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE10_M_deleterEv,"axG",@progbits,_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE10_M_deleterEv,comdat
	.align 2
	.weak	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE10_M_deleterEv
	.type	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE10_M_deleterEv, @function
_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE10_M_deleterEv:
.LFB1627:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt3getILm1EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1627:
	.size	_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE10_M_deleterEv, .-_ZNSt15__uniq_ptr_implIiSt14default_deleteIA_iEE10_M_deleterEv
	.section	.text._ZSt3getILm0EJPiSt14default_deleteIA_iEEERKNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERKS8_,"axG",@progbits,_ZSt3getILm0EJPiSt14default_deleteIA_iEEERKNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERKS8_,comdat
	.weak	_ZSt3getILm0EJPiSt14default_deleteIA_iEEERKNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERKS8_
	.type	_ZSt3getILm0EJPiSt14default_deleteIA_iEEERKNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERKS8_, @function
_ZSt3getILm0EJPiSt14default_deleteIA_iEEERKNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERKS8_:
.LFB1628:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERKT0_RKSt11_Tuple_implIXT_EJS4_DpT1_EE
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1628:
	.size	_ZSt3getILm0EJPiSt14default_deleteIA_iEEERKNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERKS8_, .-_ZSt3getILm0EJPiSt14default_deleteIA_iEEERKNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERKS8_
	.section	.text._ZNSt5tupleIJPiSt14default_deleteIA_iEEEC2ILb1ELb1EEEv,"axG",@progbits,_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC5ILb1ELb1EEEv,comdat
	.align 2
	.weak	_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC2ILb1ELb1EEEv
	.type	_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC2ILb1ELb1EEEv, @function
_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC2ILb1ELb1EEEv:
.LFB1630:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA1630
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1630:
	.section	.gcc_except_table
.LLSDA1630:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE1630-.LLSDACSB1630
.LLSDACSB1630:
.LLSDACSE1630:
	.section	.text._ZNSt5tupleIJPiSt14default_deleteIA_iEEEC2ILb1ELb1EEEv,"axG",@progbits,_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC5ILb1ELb1EEEv,comdat
	.size	_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC2ILb1ELb1EEEv, .-_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC2ILb1ELb1EEEv
	.weak	_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC1ILb1ELb1EEEv
	.set	_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC1ILb1ELb1EEEv,_ZNSt5tupleIJPiSt14default_deleteIA_iEEEC2ILb1ELb1EEEv
	.section	.text._ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE,"axG",@progbits,_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE,comdat
	.weak	_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE
	.type	_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE, @function
_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE:
.LFB1632:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERS4_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1632:
	.size	_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE, .-_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE
	.section	.text._ZSt3getILm1EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_,"axG",@progbits,_ZSt3getILm1EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_,comdat
	.weak	_ZSt3getILm1EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_
	.type	_ZSt3getILm1EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_, @function
_ZSt3getILm1EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_:
.LFB1633:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__get_helperILm1ESt14default_deleteIA_iEJEERT0_RSt11_Tuple_implIXT_EJS3_DpT1_EE
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1633:
	.size	_ZSt3getILm1EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_, .-_ZSt3getILm1EJPiSt14default_deleteIA_iEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_
	.section	.text._ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERKT0_RKSt11_Tuple_implIXT_EJS4_DpT1_EE,"axG",@progbits,_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERKT0_RKSt11_Tuple_implIXT_EJS4_DpT1_EE,comdat
	.weak	_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERKT0_RKSt11_Tuple_implIXT_EJS4_DpT1_EE
	.type	_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERKT0_RKSt11_Tuple_implIXT_EJS4_DpT1_EE, @function
_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERKT0_RKSt11_Tuple_implIXT_EJS4_DpT1_EE:
.LFB1634:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERKS4_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1634:
	.size	_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERKT0_RKSt11_Tuple_implIXT_EJS4_DpT1_EE, .-_ZSt12__get_helperILm0EPiJSt14default_deleteIA_iEEERKT0_RKSt11_Tuple_implIXT_EJS4_DpT1_EE
	.section	.text._ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC2Ev,"axG",@progbits,_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC5Ev,comdat
	.align 2
	.weak	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC2Ev
	.type	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC2Ev, @function
_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC2Ev:
.LFB1636:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC2Ev
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm0EPiLb0EEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1636:
	.size	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC2Ev, .-_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC2Ev
	.weak	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC1Ev
	.set	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC1Ev,_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEEC2Ev
	.section	.text._ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERS4_,"axG",@progbits,_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERS4_,comdat
	.weak	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERS4_
	.type	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERS4_, @function
_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERS4_:
.LFB1638:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERS1_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1638:
	.size	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERS4_, .-_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERS4_
	.section	.text._ZSt12__get_helperILm1ESt14default_deleteIA_iEJEERT0_RSt11_Tuple_implIXT_EJS3_DpT1_EE,"axG",@progbits,_ZSt12__get_helperILm1ESt14default_deleteIA_iEJEERT0_RSt11_Tuple_implIXT_EJS3_DpT1_EE,comdat
	.weak	_ZSt12__get_helperILm1ESt14default_deleteIA_iEJEERT0_RSt11_Tuple_implIXT_EJS3_DpT1_EE
	.type	_ZSt12__get_helperILm1ESt14default_deleteIA_iEJEERT0_RSt11_Tuple_implIXT_EJS3_DpT1_EE, @function
_ZSt12__get_helperILm1ESt14default_deleteIA_iEJEERT0_RSt11_Tuple_implIXT_EJS3_DpT1_EE:
.LFB1639:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEE7_M_headERS3_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1639:
	.size	_ZSt12__get_helperILm1ESt14default_deleteIA_iEJEERT0_RSt11_Tuple_implIXT_EJS3_DpT1_EE, .-_ZSt12__get_helperILm1ESt14default_deleteIA_iEJEERT0_RSt11_Tuple_implIXT_EJS3_DpT1_EE
	.section	.text._ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERKS4_,"axG",@progbits,_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERKS4_,comdat
	.weak	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERKS4_
	.type	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERKS4_, @function
_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERKS4_:
.LFB1640:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERKS1_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1640:
	.size	_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERKS4_, .-_ZNSt11_Tuple_implILm0EJPiSt14default_deleteIA_iEEE7_M_headERKS4_
	.section	.text._ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC2Ev,"axG",@progbits,_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC5Ev,comdat
	.align 2
	.weak	_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC2Ev
	.type	_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC2Ev, @function
_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC2Ev:
.LFB1642:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1642:
	.size	_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC2Ev, .-_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC2Ev
	.weak	_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC1Ev
	.set	_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC1Ev,_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEEC2Ev
	.section	.text._ZNSt10_Head_baseILm0EPiLb0EEC2Ev,"axG",@progbits,_ZNSt10_Head_baseILm0EPiLb0EEC5Ev,comdat
	.align 2
	.weak	_ZNSt10_Head_baseILm0EPiLb0EEC2Ev
	.type	_ZNSt10_Head_baseILm0EPiLb0EEC2Ev, @function
_ZNSt10_Head_baseILm0EPiLb0EEC2Ev:
.LFB1645:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	$0, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1645:
	.size	_ZNSt10_Head_baseILm0EPiLb0EEC2Ev, .-_ZNSt10_Head_baseILm0EPiLb0EEC2Ev
	.weak	_ZNSt10_Head_baseILm0EPiLb0EEC1Ev
	.set	_ZNSt10_Head_baseILm0EPiLb0EEC1Ev,_ZNSt10_Head_baseILm0EPiLb0EEC2Ev
	.section	.text._ZNSt10_Head_baseILm0EPiLb0EE7_M_headERS1_,"axG",@progbits,_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERS1_,comdat
	.weak	_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERS1_
	.type	_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERS1_, @function
_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERS1_:
.LFB1647:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1647:
	.size	_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERS1_, .-_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERS1_
	.section	.text._ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEE7_M_headERS3_,"axG",@progbits,_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEE7_M_headERS3_,comdat
	.weak	_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEE7_M_headERS3_
	.type	_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEE7_M_headERS3_, @function
_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEE7_M_headERS3_:
.LFB1648:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EE7_M_headERS3_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1648:
	.size	_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEE7_M_headERS3_, .-_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_iEEE7_M_headERS3_
	.section	.text._ZNSt10_Head_baseILm0EPiLb0EE7_M_headERKS1_,"axG",@progbits,_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERKS1_,comdat
	.weak	_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERKS1_
	.type	_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERKS1_, @function
_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERKS1_:
.LFB1649:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1649:
	.size	_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERKS1_, .-_ZNSt10_Head_baseILm0EPiLb0EE7_M_headERKS1_
	.section	.text._ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC2Ev,"axG",@progbits,_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC5Ev,comdat
	.align 2
	.weak	_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC2Ev
	.type	_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC2Ev, @function
_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC2Ev:
.LFB1651:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1651:
	.size	_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC2Ev, .-_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC2Ev
	.weak	_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC1Ev
	.set	_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC1Ev,_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EEC2Ev
	.section	.text._ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EE7_M_headERS3_,"axG",@progbits,_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EE7_M_headERS3_,comdat
	.weak	_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EE7_M_headERS3_
	.type	_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EE7_M_headERS3_, @function
_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EE7_M_headERS3_:
.LFB1653:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1653:
	.size	_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EE7_M_headERS3_, .-_ZNSt10_Head_baseILm1ESt14default_deleteIA_iELb1EE7_M_headERS3_
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
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
