'use client'

import { useTour } from '@reactour/tour'
import React from 'react'
import { Navbar, Nav } from 'react-bootstrap'
import Link from 'next/link'

function Navigation() {
    const { setIsOpen } = useTour()
    return (
        <Navbar bg="light">
            <Link href="/" style={{ marginLeft: "30px", textDecoration: "none" }}>
                <Navbar.Brand>ChannelExplorer</Navbar.Brand>
            </Link>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
                <Nav style={{
                    marginLeft: "auto",
                    marginRight: "30px"
                }}>
                    <Nav.Link className='tutorial-tutorial' onClick={() => setIsOpen(true)}>Tutorial</Nav.Link>
                    <Nav.Link as={Link} href="/about">About</Nav.Link>
                </Nav>
            </Navbar.Collapse>
        </Navbar>
    )
}

export default Navigation
